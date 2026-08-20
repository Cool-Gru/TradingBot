import asyncio
import csv
import json
import os
import time

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests

from dotenv import load_dotenv

from alpaca.data.live import NewsDataStream
from alpaca.data.historical import StockHistoricalDataClient

from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
)

from alpaca.data.timeframe import TimeFrame

from alpaca.data.enums import (
    DataFeed,
    Adjustment,
)

from alpaca.trading.client import TradingClient

from alpaca.trading.requests import (
    LimitOrderRequest,
    GetCalendarRequest,
)

from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
)

from news.news_scanner import process_article

from news.build_spike_dataset import (
    build_stock_features,
    build_market_features,
    add_relative_features,
    get_pre_event_bars,
)

from live.risk_manager import can_open_trade

from live.position_manager import record_entry


# ============================================================
# SAFETY SETTINGS
# ============================================================

# Leave this True for the first several tests.
DRY_RUN = True

# Alpaca is also hard-coded to paper mode below.
PAPER_MODE = True

MODEL_PATH = "news/spike_model_v5.joblib"

LOG_PATH = "live/realtime_news_log.csv"

SEEN_NEWS_PATH = "live/realtime_seen_news.json"


# ============================================================
# SIGNAL REQUIREMENTS
# ============================================================

# Immediate-entry behavior was not part of the original
# validation, so start stricter than the normal V5 scanner.
MIN_SPIKE_SCORE = 0.75

MIN_SENTIMENT = 2

MIN_IMPORTANCE = 5

MAX_ARTICLE_AGE_SECONDS = 180

# On startup, recover news that arrived while the program was offline. The
# longer window covers weekends and most market holidays; only still-valid
# next-session candidates are shown as watches, never submitted as orders.
BACKFILL_LOOKBACK_HOURS = int(
    os.getenv("NEWS_BACKFILL_HOURS", "96")
)
BACKFILL_MAX_ARTICLES = int(
    os.getenv("NEWS_BACKFILL_MAX_ARTICLES", "500")
)
BACKFILL_PAGE_SIZE = 50
NEXT_ENTRY_WINDOW_MINUTES = 30

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"


# ============================================================
# MARKET SAFETY FILTERS
# ============================================================

MIN_SHARE_PRICE = 5.00

# Based on prior 20 completed sessions.
MIN_AVG_DOLLAR_VOLUME = 1_000_000

# Do not chase a stock that already moved more than 4%
# from its previous completed close.
MAX_MOVE_FROM_PREVIOUS_CLOSE = 0.04

# Reject extremely wide bid/ask spreads.
MAX_SPREAD_PERCENT = 0.015

# Limit price can be at most 0.30% above the current ask.
LIMIT_PRICE_BUFFER = 0.003


# ============================================================
# POSITION SIZING
# ============================================================

# Start immediate-news paper trades at only 1% of equity.
NEWS_POSITION_PERCENT = 0.01

BUDGET_BUFFER = 0.98


# ============================================================
# ORDER SETTINGS
# ============================================================

ORDER_FILL_TIMEOUT_SECONDS = 30

ORDER_CHECK_INTERVAL_SECONDS = 1


# ============================================================
# DATA SETTINGS
# ============================================================

PRICE_LOOKBACK_DAYS = 450

NEW_YORK = ZoneInfo("America/New_York")


# ============================================================
# API CONNECTIONS
# ============================================================

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")


if not API_KEY or not SECRET_KEY:

    raise RuntimeError(
        "Missing ALPACA_API_KEY or "
        "ALPACA_SECRET_KEY in .env"
    )


trading_client = TradingClient(
    API_KEY,
    SECRET_KEY,
    paper=PAPER_MODE,
)


stock_client = StockHistoricalDataClient(
    API_KEY,
    SECRET_KEY,
)


news_stream = NewsDataStream(
    API_KEY,
    SECRET_KEY,
)


# ============================================================
# LOAD V5 MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Missing {MODEL_PATH}. "
        "Make sure V5 training completed."
    )


model_bundle = joblib.load(
    MODEL_PATH
)


model = model_bundle["model"]

FEATURE_COLUMNS = model_bundle["features"]

NUMERIC_COLUMNS = model_bundle[
    "numeric_features"
]


# ============================================================
# CACHE
# ============================================================

history_cache = {}

market_cache = {}

processing_lock = None

calendar_cache = {}

processed_this_run = set()

log_schema_checked = False


# ============================================================
# SEEN ARTICLE STORAGE
# ============================================================

def load_seen_news():

    if not os.path.exists(SEEN_NEWS_PATH):
        return set()

    try:

        with open(
            SEEN_NEWS_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            values = json.load(file)

        return set(values)

    except Exception:

        return set()


seen_news = load_seen_news()


def save_seen_news():

    # Keep only the most recent portion so the file
    # does not grow forever.
    values = list(seen_news)[-5000:]

    Path(SEEN_NEWS_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        SEEN_NEWS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            values,
            file,
            indent=2,
        )


def build_seen_key(article):

    article_id = article.get(
        "id",
        "unknown",
    )

    updated_at = article.get(
        "updated_at",
        "",
    )

    return (
        f"{article_id}:"
        f"{updated_at}"
    )


# ============================================================
# CSV LOG
# ============================================================

LOG_FIELDS = [
    "processed_at",
    "article_id",
    "article_time",
    "symbol",
    "headline",
    "event_type",
    "sentiment",
    "importance",
    "spike_score",
    "session",
    "current_price",
    "previous_close",
    "move_from_previous_close",
    "spread_percent",
    "avg_dollar_volume_20",
    "decision",
    "details",
    "source_mode",
    "expires_at",
]


def append_log(**values):

    global log_schema_checked

    Path(LOG_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = os.path.exists(LOG_PATH)

    # Upgrade an older log header before appending the new startup-status
    # fields. This preserves every existing row and keeps pandas parsing clean.
    if exists and not log_schema_checked:
        try:
            with open(LOG_PATH, "r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                existing_fields = reader.fieldnames or []
                existing_rows = list(reader)

            if existing_fields != LOG_FIELDS:
                temporary_path = f"{LOG_PATH}.tmp"
                with open(
                    temporary_path,
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as file:
                    writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)
                    writer.writeheader()
                    for existing_row in existing_rows:
                        writer.writerow({
                            field: existing_row.get(field, "")
                            for field in LOG_FIELDS
                        })
                os.replace(temporary_path, LOG_PATH)
        except Exception as error:
            print(f"Warning: could not upgrade signal log schema: {error}")

    log_schema_checked = True

    row = {
        field: values.get(
            field,
            "",
        )
        for field in LOG_FIELDS
    }

    with open(
        LOG_PATH,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=LOG_FIELDS,
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# NEWS OBJECT TO DICTIONARY
# ============================================================

def news_to_dict(article):

    if isinstance(article, dict):
        return article

    return {
        "id":
            article.id,

        "headline":
            article.headline,

        "summary":
            article.summary,

        "content":
            article.content,

        "source":
            article.source,

        "author":
            article.author,

        "url":
            article.url,

        "symbols":
            list(article.symbols),

        "created_at":
            article.created_at.isoformat(),

        "updated_at":
            article.updated_at.isoformat(),
    }


# ============================================================
# STARTUP NEWS CATCH-UP
# ============================================================

def fetch_startup_news():

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=BACKFILL_LOOKBACK_HOURS)

    headers = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": SECRET_KEY,
    }

    articles = []
    page_token = None

    while len(articles) < BACKFILL_MAX_ARTICLES:

        params = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": min(
                BACKFILL_PAGE_SIZE,
                BACKFILL_MAX_ARTICLES - len(articles),
            ),
            # Pull newest pages first so a capped catch-up never discards the
            # most relevant articles.
            "sort": "desc",
            "include_content": "true",
        }

        if page_token:
            params["page_token"] = page_token

        response = requests.get(
            NEWS_URL,
            headers=headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()

        payload = response.json()
        page = payload.get("news", [])

        if not page:
            break

        articles.extend(page)
        page_token = payload.get("next_page_token")

        if not page_token:
            break

    # The API may return revised copies. Retain the newest copy of each exact
    # article/update pair, then process in publication order.
    unique = {}
    for article in articles:
        unique[build_seen_key(article)] = article

    def article_time(article):
        return pd.to_datetime(
            article.get("updated_at") or article.get("created_at"),
            utc=True,
            errors="coerce",
        )

    return sorted(
        unique.values(),
        key=lambda article: (
            article_time(article)
            if not pd.isna(article_time(article))
            else pd.Timestamp.min.tz_localize("UTC")
        ),
    )


def next_regular_open_after(event_time):

    event_et = localize_market_time(event_time)
    start_date = event_et.date()
    end_date = start_date + timedelta(days=10)
    cache_key = (start_date, end_date)

    if cache_key not in calendar_cache:
        request = GetCalendarRequest(start=start_date, end=end_date)
        calendar_cache[cache_key] = trading_client.get_calendar(
            filters=request
        )

    for session in calendar_cache[cache_key]:
        market_open = localize_market_time(session.open)
        if market_open > event_et:
            return market_open

    return None


def startup_signal_expiration(event_time):

    try:
        next_open = next_regular_open_after(event_time)
    except Exception as error:
        print(f"Calendar lookup failed: {error}")
        next_open = None

    if next_open is None:
        return localize_market_time(event_time) + timedelta(hours=24)

    return next_open + timedelta(minutes=NEXT_ENTRY_WINDOW_MINUTES)


def run_startup_backfill():

    print(
        f"\nChecking the last {BACKFILL_LOOKBACK_HOURS} hours "
        "for missed news..."
    )

    try:
        articles = fetch_startup_news()
    except Exception as error:
        print(f"Startup news catch-up failed: {error}")
        print("Continuing with the live stream.")
        return

    print(f"Startup articles received: {len(articles)}")

    for article in articles:
        try:
            process_news_sync(article, source_mode="BACKFILL")
        except Exception as error:
            article_id = article.get("id", "unknown")
            print(f"Backfill article {article_id} failed: {error}")

    save_seen_news()
    print("Startup catch-up complete.")


# ============================================================
# MARKET SESSION
# ============================================================

def localize_market_time(value):

    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:

        timestamp = timestamp.tz_localize(
            NEW_YORK
        )

    else:

        timestamp = timestamp.tz_convert(
            NEW_YORK
        )

    return timestamp


def get_current_session():

    clock = trading_client.get_clock()

    now_et = localize_market_time(
        clock.timestamp
    )

    if clock.is_open:

        return {
            "name": "REGULAR",
            "tradable": True,
            "extended_hours": False,
            "now": now_et,
        }

    today = now_et.date()

    request = GetCalendarRequest(
        start=today,
        end=today,
    )

    calendar = trading_client.get_calendar(
        filters=request
    )

    if not calendar:

        return {
            "name": "CLOSED",
            "tradable": False,
            "extended_hours": False,
            "now": now_et,
        }

    session = calendar[0]

    market_open = localize_market_time(
        session.open
    )

    market_close = localize_market_time(
        session.close
    )

    premarket_start = market_open.replace(
        hour=4,
        minute=0,
        second=0,
        microsecond=0,
    )

    after_hours_end = market_close.replace(
        hour=20,
        minute=0,
        second=0,
        microsecond=0,
    )

    if (
        premarket_start
        <= now_et
        < market_open
    ):

        return {
            "name": "PREMARKET",
            "tradable": True,
            "extended_hours": True,
            "now": now_et,
        }

    if (
        market_close
        < now_et
        <= after_hours_end
    ):

        return {
            "name": "AFTER_HOURS",
            "tradable": True,
            "extended_hours": True,
            "now": now_et,
        }

    return {
        "name": "CLOSED",
        "tradable": False,
        "extended_hours": False,
        "now": now_et,
    }


# ============================================================
# PRICE HISTORY
# ============================================================

def download_history(
    symbol,
    event_time,
):

    event_timestamp = pd.to_datetime(
        event_time,
        utc=True,
    )

    event_date = (
        event_timestamp
        .tz_convert(NEW_YORK)
        .date()
    )

    cache_key = (
        symbol,
        event_date,
    )

    if cache_key in history_cache:

        return history_cache[
            cache_key
        ]

    end = datetime.now(
        timezone.utc
    )

    start = (
        end
        - timedelta(
            days=PRICE_LOOKBACK_DAYS
        )
    )

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
        adjustment=Adjustment.ALL,
    )

    bars = stock_client.get_stock_bars(
        request
    )

    df = bars.df.reset_index()

    if df.empty:

        history_cache[
            cache_key
        ] = None

        return None

    if "symbol" in df.columns:

        df = (
            df[
                df["symbol"]
                == symbol
            ]
            .copy()
        )

    if df.empty:

        history_cache[
            cache_key
        ] = None

        return None

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df["session_date"] = (
        df["timestamp"]
        .dt.tz_convert(
            NEW_YORK
        )
        .dt.date
    )

    df = (
        df
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    history_cache[
        cache_key
    ] = df

    return df


def get_market_history(
    event_time,
):

    event_timestamp = pd.to_datetime(
        event_time,
        utc=True,
    )

    event_date = (
        event_timestamp
        .tz_convert(NEW_YORK)
        .date()
    )

    if event_date in market_cache:

        return market_cache[
            event_date
        ]

    spy_df = download_history(
        "SPY",
        event_time,
    )

    qqq_df = download_history(
        "QQQ",
        event_time,
    )

    if (
        spy_df is None
        or
        qqq_df is None
    ):

        return None

    market_cache[
        event_date
    ] = (
        spy_df,
        qqq_df,
    )

    return market_cache[
        event_date
    ]


# ============================================================
# QUOTE
# ============================================================

def get_latest_quote(
    symbol
):

    request = StockLatestQuoteRequest(
        symbol_or_symbols=symbol,
        feed=DataFeed.IEX,
    )

    quotes = (
        stock_client
        .get_stock_latest_quote(
            request
        )
    )

    quote = quotes.get(
        symbol
    )

    if quote is None:
        return None

    bid = float(
        quote.bid_price
    )

    ask = float(
        quote.ask_price
    )

    if (
        bid <= 0
        or
        ask <= 0
        or
        ask < bid
    ):

        return None

    midpoint = (
        bid
        + ask
    ) / 2

    spread_percent = (
        ask
        - bid
    ) / midpoint

    return {
        "bid": bid,
        "ask": ask,
        "midpoint": midpoint,
        "spread_percent":
            spread_percent,
    }


# ============================================================
# SCORE ONE SYMBOL
# ============================================================

def score_symbol(
    article,
    symbol,
):

    try:

        asset = trading_client.get_asset(
            symbol
        )

    except Exception:

        return None

    if not asset.tradable:
        return None

    event_time = article[
        "event_time"
    ]

    stock_df = download_history(
        symbol,
        event_time,
    )

    if stock_df is None:
        return None

    pre = get_pre_event_bars(
        stock_df,
        event_time,
    )

    if len(pre) < 20:
        return None

    previous_close = float(
        pre.iloc[-1]["close"]
    )

    if previous_close < MIN_SHARE_PRICE:
        return None

    average_dollar_volume = float(
        (
            pre.tail(20)["close"]
            * pre.tail(20)["volume"]
        ).mean()
    )

    if (
        not np.isfinite(
            average_dollar_volume
        )
        or
        average_dollar_volume
        < MIN_AVG_DOLLAR_VOLUME
    ):

        return None

    market_history = get_market_history(
        event_time
    )

    if market_history is None:
        return None

    spy_df, qqq_df = market_history

    event = dict(article)

    event["symbol"] = symbol

    row = {
        "text":
            (
                str(
                    article.get(
                        "headline",
                        "",
                    )
                )
                + " "
                + str(
                    article.get(
                        "summary",
                        "",
                    )
                )
            ),

        "event_type":
            article.get(
                "event_type",
                "OTHER",
            ),

        "event_score":
            article.get(
                "event_score",
                0,
            ),

        "sentiment":
            article.get(
                "sentiment",
                0,
            ),

        "future_score":
            article.get(
                "future_score",
                0,
            ),

        "importance":
            article.get(
                "importance",
                0,
            ),
    }

    row.update(
        build_stock_features(
            event,
            stock_df,
        )
    )

    row.update(
        build_market_features(
            event_time,
            spy_df,
            qqq_df,
        )
    )

    row = add_relative_features(
        row
    )

    for column in NUMERIC_COLUMNS:

        if column not in row:

            row[column] = np.nan

    X = pd.DataFrame(
        [
            {
                column:
                    row.get(
                        column,
                        np.nan,
                    )

                for column
                in FEATURE_COLUMNS
            }
        ]
    )

    probability = float(
        model.predict_proba(
            X
        )[0][1]
    )

    return {
        "symbol": symbol,
        "score": probability,
        "previous_close":
            previous_close,
        "avg_dollar_volume_20":
            average_dollar_volume,
    }


# ============================================================
# WAIT FOR FILL
# ============================================================

def wait_for_fill(
    order_id
):

    start_time = time.time()

    latest_order = None

    while (
        time.time()
        - start_time
        < ORDER_FILL_TIMEOUT_SECONDS
    ):

        latest_order = (
            trading_client
            .get_order_by_id(
                order_id
            )
        )

        filled_qty = float(
            latest_order.filled_qty
            or 0
        )

        if (
            filled_qty > 0
            and
            latest_order
            .filled_avg_price
            is not None
        ):

            return latest_order

        time.sleep(
            ORDER_CHECK_INTERVAL_SECONDS
        )

    try:

        trading_client.cancel_order_by_id(
            order_id
        )

    except Exception:

        pass

    # Check whether a partial fill occurred.
    try:

        latest_order = (
            trading_client
            .get_order_by_id(
                order_id
            )
        )

    except Exception:

        return None

    filled_qty = float(
        latest_order.filled_qty
        or 0
    )

    if (
        filled_qty > 0
        and
        latest_order
        .filled_avg_price
        is not None
    ):

        return latest_order

    return None


# ============================================================
# PLACE PAPER ORDER
# ============================================================

def attempt_order(
    article,
    candidate,
    quote,
    session,
):

    symbol = candidate[
        "symbol"
    ]

    risk = can_open_trade(
        symbol
    )

    if not risk["approved"]:

        details = "; ".join(
            risk["reasons"]
        )

        print(
            f"  Risk manager rejected: "
            f"{details}"
        )

        append_log(
            processed_at=
                datetime.now(
                    timezone.utc
                ).isoformat(),

            article_id=
                article.get(
                    "article_id"
                ),

            article_time=
                article.get(
                    "event_time"
                ),

            symbol=
                symbol,

            headline=
                article.get(
                    "headline"
                ),

            event_type=
                article.get(
                    "event_type"
                ),

            sentiment=
                article.get(
                    "sentiment"
                ),

            importance=
                article.get(
                    "importance"
                ),

            spike_score=
                candidate["score"],

            session=
                session["name"],

            decision=
                "RISK_REJECTED",

            details=
                details,
        )

        return

    equity = risk[
        "account"
    ][
        "equity"
    ]

    cash = risk[
        "account"
    ][
        "cash"
    ]

    immediate_limit = (
        equity
        * NEWS_POSITION_PERCENT
    )

    budget = min(
        immediate_limit,
        risk["budget"],
        cash,
    )

    usable_budget = (
        budget
        * BUDGET_BUFFER
    )

    limit_price = round(
        quote["ask"]
        * (
            1
            + LIMIT_PRICE_BUFFER
        ),
        2,
    )

    shares = int(
        usable_budget
        // limit_price
    )

    if shares <= 0:

        print(
            "  Insufficient budget "
            "for one share."
        )

        return

    estimated_value = (
        shares
        * limit_price
    )

    print(
        "\n=========================================="
    )

    print(
        "       IMMEDIATE NEWS TRADE SIGNAL"
    )

    print(
        "=========================================="
    )

    print(
        f"\nSymbol: "
        f"{symbol}"
    )

    print(
        f"V5 score: "
        f"{candidate['score']:.2%}"
    )

    print(
        f"Sentiment: "
        f"{article['sentiment']}"
    )

    print(
        f"Session: "
        f"{session['name']}"
    )

    print(
        f"Ask price: "
        f"${quote['ask']:.2f}"
    )

    print(
        f"Limit price: "
        f"${limit_price:.2f}"
    )

    print(
        f"Shares: "
        f"{shares}"
    )

    print(
        f"Estimated value: "
        f"${estimated_value:,.2f}"
    )

    if DRY_RUN:

        print(
            "\nDRY RUN: "
            "NO ORDER SUBMITTED."
        )

        append_log(
            processed_at=
                datetime.now(
                    timezone.utc
                ).isoformat(),

            article_id=
                article.get(
                    "article_id"
                ),

            article_time=
                article.get(
                    "event_time"
                ),

            symbol=
                symbol,

            headline=
                article.get(
                    "headline"
                ),

            event_type=
                article.get(
                    "event_type"
                ),

            sentiment=
                article.get(
                    "sentiment"
                ),

            importance=
                article.get(
                    "importance"
                ),

            spike_score=
                candidate["score"],

            session=
                session["name"],

            current_price=
                quote["ask"],

            previous_close=
                candidate[
                    "previous_close"
                ],

            move_from_previous_close=
                (
                    quote["ask"]
                    / candidate[
                        "previous_close"
                    ]
                    - 1
                ),

            spread_percent=
                quote[
                    "spread_percent"
                ],

            avg_dollar_volume_20=
                candidate[
                    "avg_dollar_volume_20"
                ],

            decision=
                "DRY_RUN_BUY",

            details=
                (
                    f"Would buy "
                    f"{shares} shares "
                    f"at limit "
                    f"${limit_price:.2f}"
                ),
        )

        return

    client_order_id = (
        "news-"
        + str(
            article.get(
                "article_id",
                "unknown",
            )
        )
        + "-"
        + symbol
        + "-"
        + str(
            int(
                time.time()
            )
        )
    )[:48]

    order_request = LimitOrderRequest(
        symbol=symbol,
        qty=shares,
        side=OrderSide.BUY,
        limit_price=limit_price,
        time_in_force=TimeInForce.DAY,
        extended_hours=
            session[
                "extended_hours"
            ],
        client_order_id=
            client_order_id,
    )

    print(
        "\nSubmitting PAPER "
        "limit order..."
    )

    order = trading_client.submit_order(
        order_data=order_request
    )

    filled_order = wait_for_fill(
        order.id
    )

    if filled_order is None:

        print(
            "\nOrder was not filled "
            "and was canceled."
        )

        append_log(
            processed_at=
                datetime.now(
                    timezone.utc
                ).isoformat(),

            article_id=
                article.get(
                    "article_id"
                ),

            article_time=
                article.get(
                    "event_time"
                ),

            symbol=
                symbol,

            headline=
                article.get(
                    "headline"
                ),

            event_type=
                article.get(
                    "event_type"
                ),

            sentiment=
                article.get(
                    "sentiment"
                ),

            importance=
                article.get(
                    "importance"
                ),

            spike_score=
                candidate["score"],

            session=
                session["name"],

            current_price=
                quote["ask"],

            spread_percent=
                quote[
                    "spread_percent"
                ],

            decision=
                "ORDER_NOT_FILLED",

            details=
                f"Order ID {order.id}",
        )

        return

    fill_price = float(
        filled_order
        .filled_avg_price
    )

    filled_qty = float(
        filled_order
        .filled_qty
    )

    record_entry(
        symbol=symbol,
        entry_price=fill_price,
        quantity=filled_qty,
        probability=
            candidate["score"],
    )

    print(
        "\n=========================================="
    )

    print(
        "       PAPER NEWS ORDER FILLED"
    )

    print(
        "=========================================="
    )

    print(
        f"\nSymbol: "
        f"{symbol}"
    )

    print(
        f"Quantity: "
        f"{filled_qty}"
    )

    print(
        f"Fill price: "
        f"${fill_price:.2f}"
    )

    append_log(
        processed_at=
            datetime.now(
                timezone.utc
            ).isoformat(),

        article_id=
            article.get(
                "article_id"
            ),

        article_time=
            article.get(
                "event_time"
            ),

        symbol=
            symbol,

        headline=
            article.get(
                "headline"
            ),

        event_type=
            article.get(
                "event_type"
            ),

        sentiment=
            article.get(
                "sentiment"
            ),

        importance=
            article.get(
                "importance"
            ),

        spike_score=
            candidate["score"],

        session=
            session["name"],

        current_price=
            fill_price,

        previous_close=
            candidate[
                "previous_close"
            ],

        move_from_previous_close=
            (
                fill_price
                / candidate[
                    "previous_close"
                ]
                - 1
            ),

        spread_percent=
            quote[
                "spread_percent"
            ],

        avg_dollar_volume_20=
            candidate[
                "avg_dollar_volume_20"
            ],

        decision=
            "PAPER_BUY_FILLED",

        details=
            (
                f"{filled_qty} shares; "
                f"order {filled_order.id}"
            ),
    )


# ============================================================
# PROCESS ONE NEWS ARTICLE
# ============================================================

def append_candidate_status(
    article,
    candidate,
    quote,
    session,
    decision,
    details,
    source_mode,
    expires_at=None,
):

    append_log(
        processed_at=datetime.now(timezone.utc).isoformat(),
        article_id=article.get("article_id"),
        article_time=article.get("event_time"),
        symbol=candidate["symbol"],
        headline=article.get("headline"),
        event_type=article.get("event_type"),
        sentiment=article.get("sentiment"),
        importance=article.get("importance"),
        spike_score=candidate["score"],
        session=session["name"],
        current_price=quote["ask"],
        previous_close=candidate["previous_close"],
        move_from_previous_close=(
            quote["ask"] / candidate["previous_close"] - 1
        ),
        spread_percent=quote["spread_percent"],
        avg_dollar_volume_20=candidate["avg_dollar_volume_20"],
        decision=decision,
        details=details,
        source_mode=source_mode,
        expires_at=(
            expires_at.isoformat()
            if expires_at is not None
            else ""
        ),
    )


def process_news_sync(
    incoming_article,
    source_mode="LIVE",
):

    raw_article = news_to_dict(
        incoming_article
    )

    seen_key = build_seen_key(
        raw_article
    )

    if seen_key in processed_this_run:
        return

    processed_this_run.add(seen_key)

    # Backfill deliberately re-scores recent articles with a fresh quote every
    # time the program starts. Live delivery still uses persistent deduping.
    if (
        source_mode == "LIVE"
        and seen_key in seen_news
    ):
        return

    seen_news.add(
        seen_key
    )

    if source_mode == "LIVE":
        save_seen_news()

    article = process_article(
        raw_article
    )

    if article is None:
        return

    headline = str(
        article.get(
            "headline",
            "",
        )
    )

    print(
        "\n=========================================="
    )

    print(
        "          NEW ARTICLE RECEIVED"
    )

    print(
        "=========================================="
    )

    print(
        f"\n{headline}"
    )

    print(
        f"Symbols: "
        f"{', '.join(article['symbols'])}"
    )

    print(
        f"Event: "
        f"{article['event_type']}"
    )

    print(
        f"Sentiment: "
        f"{article['sentiment']}"
    )

    print(
        f"Importance: "
        f"{article['importance']}"
    )

    # --------------------------------------------------------
    # ARTICLE FILTERS
    # --------------------------------------------------------

    if headline.lower().startswith(
        "correction:"
    ):

        print(
            "Rejected: correction article."
        )

        return

    if article.get(
        "post_move",
        False,
    ):

        print(
            "Rejected: article describes "
            "a move that already happened."
        )

        return

    event_time = pd.to_datetime(
        article["event_time"],
        utc=True,
    )

    age_seconds = (
        datetime.now(
            timezone.utc
        )
        - event_time.to_pydatetime()
    ).total_seconds()

    if age_seconds < 0:
        age_seconds = 0

    if (
        age_seconds
        > MAX_ARTICLE_AGE_SECONDS
        and source_mode == "LIVE"
    ):

        print(
            f"Rejected: article is "
            f"{age_seconds:.0f}s old."
        )

        return

    if (
        article["sentiment"]
        < MIN_SENTIMENT
    ):

        print(
            "Rejected: sentiment is "
            "not positive enough."
        )

        return

    if (
        article["importance"]
        < MIN_IMPORTANCE
    ):

        print(
            "Rejected: importance is "
            "too low."
        )

        return

    # --------------------------------------------------------
    # TRADING SESSION
    # --------------------------------------------------------

    session = get_current_session()

    print(
        f"Session: "
        f"{session['name']}"
    )

    # --------------------------------------------------------
    # SCORE EVERY SYMBOL IN THE ARTICLE
    # --------------------------------------------------------

    scored = []

    for symbol in article[
        "symbols"
    ]:

        print(
            f"\nScoring "
            f"{symbol}..."
        )

        try:

            result = score_symbol(
                article,
                symbol,
            )

        except Exception as error:

            print(
                f"  Score failed: "
                f"{error}"
            )

            continue

        if result is None:

            print(
                "  Skipped by data or "
                "liquidity filters."
            )

            continue

        print(
            f"  V5 score: "
            f"{result['score']:.2%}"
        )

        scored.append(
            result
        )

    if not scored:

        return

    scored.sort(
        key=lambda value:
            value["score"],
        reverse=True,
    )

    candidate = scored[0]

    symbol = candidate[
        "symbol"
    ]

    if (
        candidate["score"]
        < MIN_SPIKE_SCORE
    ):

        print(
            f"\nBest score "
            f"{candidate['score']:.2%} "
            f"is below "
            f"{MIN_SPIKE_SCORE:.0%}."
        )

        return

    # --------------------------------------------------------
    # CURRENT QUOTE
    # --------------------------------------------------------

    quote = get_latest_quote(
        symbol
    )

    if quote is None:

        print(
            "Rejected: no valid quote."
        )

        return

    if (
        quote["ask"]
        < MIN_SHARE_PRICE
    ):

        print(
            "Rejected: share price "
            "below minimum."
        )

        return

    if (
        quote[
            "spread_percent"
        ]
        > MAX_SPREAD_PERCENT
    ):

        print(
            f"Rejected: spread is "
            f"{quote['spread_percent']:.2%}."
        )

        return

    move_from_close = (
        quote["ask"]
        / candidate[
            "previous_close"
        ]
        - 1
    )

    print(
        f"Current move from "
        f"previous close: "
        f"{move_from_close:.2%}"
    )

    if (
        move_from_close
        > MAX_MOVE_FROM_PREVIOUS_CLOSE
    ):

        print(
            "Rejected: stock already "
            "moved too far. Not chasing."
        )

        return

    # Startup articles are recommendations only. They are evaluated with a
    # fresh quote, but cannot submit a delayed order. Their next-session entry
    # window is explicit so the dashboard does not present a stale BUY.
    if source_mode == "BACKFILL":

        expires_at = startup_signal_expiration(
            article["event_time"]
        )
        now_et = session["now"]
        expires_text = expires_at.strftime(
            "%Y-%m-%d %I:%M %p ET"
        )

        if now_et > expires_at:
            decision = "BACKFILL_EXPIRED"
            details = (
                "Startup catch-up: model-qualified article, but its "
                f"next-session entry window expired at {expires_text}. "
                "No order submitted."
            )
        elif session["name"] == "PREMARKET":
            decision = "PREMARKET_BUY_WATCH"
            details = (
                "Recovered at startup and still inside the next-session "
                f"window through {expires_text}. "
                "No delayed order was submitted; recheck at the open."
            )
        elif session["name"] == "CLOSED":
            decision = "NEXT_OPEN_BUY_WATCH"
            details = (
                "Recovered at startup and model-qualified for the next "
                f"regular session; entry window ends {expires_text}. "
                "No order submitted."
            )
        else:
            decision = "STARTUP_BUY_WATCH"
            details = (
                "Recovered at startup and still inside its next-session "
                f"entry window through {expires_text}. "
                "No delayed order was submitted."
            )

        print(f"Startup decision: {decision}")
        print(details)

        append_candidate_status(
            article=article,
            candidate=candidate,
            quote=quote,
            session=session,
            decision=decision,
            details=details,
            source_mode=source_mode,
            expires_at=expires_at,
        )
        return

    if not session["tradable"]:
        print(
            "Live article qualified, but the supported trading session "
            "is closed. It will be reconsidered by startup catch-up."
        )
        return

    attempt_order(
        article,
        candidate,
        quote,
        session,
    )


# ============================================================
# ASYNC STREAM HANDLER
# ============================================================

async def handle_news(
    article
):

    global processing_lock

    if processing_lock is None:

        processing_lock = (
            asyncio.Lock()
        )

    async with processing_lock:

        try:

            await asyncio.to_thread(
                process_news_sync,
                article,
            )

        except Exception as error:

            print(
                "\nNews processing error:"
            )

            print(error)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n=========================================="
    )

    print(
        "       REAL-TIME NEWS PAPER TRADER"
    )

    print(
        "=========================================="
    )

    print(
        f"\nDRY RUN: "
        f"{DRY_RUN}"
    )

    print(
        f"Alpaca paper mode: "
        f"{PAPER_MODE}"
    )

    print(
        f"Minimum V5 score: "
        f"{MIN_SPIKE_SCORE:.0%}"
    )

    print(
        f"Position size: "
        f"{NEWS_POSITION_PERCENT:.0%} "
        f"of paper equity"
    )

    print(
        "\nSupported immediate sessions:"
    )

    print(
        "  Premarket:    4:00–9:30 AM ET"
    )

    print(
        "  Regular:     9:30 AM–4:00 PM ET"
    )

    print(
        "  After-hours: 4:00–8:00 PM ET"
    )

    print(
        "\nOvernight trading is disabled."
    )

    run_startup_backfill()

    # The catch-up set prevents the WebSocket from repeating an article that
    # arrived while the REST request was running, while future updates still
    # have a different article/update key.
    print("\nWaiting for new Alpaca news...")

    print(
        "Press Ctrl+C to stop."
    )

    news_stream.subscribe_news(
        handle_news,
        "*",
    )

    news_stream.run()
