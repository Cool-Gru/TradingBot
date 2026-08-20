import os
import re
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment

from news.news_scanner import (
    classify_event,
    sentiment_score,
    future_catalyst_score,
    is_post_move_article,
    calculate_importance,
    get_event_time,
)


# ============================================================
# SETTINGS
# ============================================================

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

OUTPUT_FILE = "news/spike_dataset_v5.csv"


# ============================================================
# PILOT PERIOD
# ============================================================

NEWS_START = datetime(
    2025,
    1,
    1,
    tzinfo=timezone.utc
)

NEWS_END = datetime(
    2025,
    4,
    1,
    tzinfo=timezone.utc
)


NEWS_CHUNK_DAYS = 7
NEWS_PAGE_LIMIT = 50

MAX_SYMBOLS_PER_ARTICLE = 4
MIN_IMPORTANCE = 2
MAX_UPDATE_AGE_HOURS = 24


# ============================================================
# PRICE SETTINGS
# ============================================================

# Need enough history for SMA200.
PRICE_LOOKBACK_DAYS = 450

# Need future data for labels.
PRICE_FORWARD_DAYS = 30

SYMBOL_BATCH_SIZE = 25

# 25 symbols * roughly 300 trading days
# stays below this for most batches.
PRICE_REQUEST_LIMIT = 10000


# ============================================================
# TARGETS
# ============================================================

SPIKE_WINDOW = 5

SPIKE_3_PERCENT = 0.03
SPIKE_5_PERCENT = 0.05
SPIKE_10_PERCENT = 0.10


# ============================================================
# TIMEZONE
# ============================================================

NEW_YORK = ZoneInfo(
    "America/New_York"
)


# ============================================================
# API
# ============================================================

load_dotenv()

API_KEY = os.getenv(
    "ALPACA_API_KEY"
)

SECRET_KEY = os.getenv(
    "ALPACA_SECRET_KEY"
)


if not API_KEY or not SECRET_KEY:
    raise RuntimeError(
        "Missing Alpaca API keys."
    )


HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY,
}


stock_client = StockHistoricalDataClient(
    API_KEY,
    SECRET_KEY
)


# ============================================================
# SYMBOL CLEANING
# ============================================================

def clean_historical_symbols(symbols):

    clean = []

    for symbol in symbols:

        symbol = (
            str(symbol)
            .upper()
            .strip()
        )

        if ":" in symbol:
            continue

        if symbol.endswith("USD"):
            continue

        if not re.fullmatch(
            r"[A-Z]{1,5}",
            symbol
        ):
            continue

        clean.append(
            symbol
        )

    return list(
        dict.fromkeys(clean)
    )


# ============================================================
# NEWS REQUEST
# ============================================================

def request_news(params):

    for attempt in range(5):

        response = requests.get(
            NEWS_URL,
            headers=HEADERS,
            params=params,
            timeout=30
        )

        if response.status_code == 429:

            wait_time = (
                5
                + attempt * 5
            )

            print(
                f"Rate limited. "
                f"Waiting {wait_time}s..."
            )

            time.sleep(
                wait_time
            )

            continue

        response.raise_for_status()

        return response.json()

    raise RuntimeError(
        "News API repeatedly rate limited."
    )


# ============================================================
# FETCH NEWS CHUNK
# ============================================================

def fetch_news_chunk(
    start,
    end
):

    articles = []

    page_token = None

    while True:

        params = {
            "start":
                start.isoformat(),

            "end":
                end.isoformat(),

            "sort":
                "asc",

            "limit":
                NEWS_PAGE_LIMIT,

            "include_content":
                "true",
        }

        if page_token:

            params[
                "page_token"
            ] = page_token

        data = request_news(
            params
        )

        page = data.get(
            "news",
            []
        )

        articles.extend(
            page
        )

        print(
            f"    +{len(page)} articles "
            f"(chunk total {len(articles)})"
        )

        page_token = data.get(
            "next_page_token"
        )

        if not page_token:
            break

    return articles


# ============================================================
# RAW ARTICLE DEDUPLICATION
# ============================================================

def deduplicate_articles(
    articles
):

    unique = {}

    duplicates = 0

    for article in articles:

        article_id = article.get(
            "id"
        )

        if article_id is not None:

            key = (
                "id",
                str(article_id)
            )

        else:

            key = (
                "fallback",
                article.get(
                    "headline",
                    ""
                ),
                article.get(
                    "created_at",
                    ""
                ),
                article.get(
                    "updated_at",
                    ""
                ),
            )

        if key in unique:

            duplicates += 1
            continue

        unique[key] = article

    print(
        f"\nRemoved duplicate articles: "
        f"{duplicates:,}"
    )

    return list(
        unique.values()
    )


# ============================================================
# FETCH ALL HISTORICAL NEWS
# ============================================================

def fetch_historical_news():

    print(
        "\n=========================================="
    )
    print(
        "       DOWNLOADING HISTORICAL NEWS"
    )
    print(
        "=========================================="
    )

    all_articles = []

    current = NEWS_START

    while current < NEWS_END:

        chunk_end = min(
            current
            + timedelta(
                days=NEWS_CHUNK_DAYS
            ),
            NEWS_END
        )

        print(
            f"\n{current.date()} "
            f"→ {chunk_end.date()}"
        )

        chunk = fetch_news_chunk(
            current,
            chunk_end
        )

        all_articles.extend(
            chunk
        )

        current = chunk_end

    print(
        f"\nRaw articles downloaded: "
        f"{len(all_articles):,}"
    )

    all_articles = deduplicate_articles(
        all_articles
    )

    print(
        f"Unique articles: "
        f"{len(all_articles):,}"
    )

    return all_articles


# ============================================================
# PROCESS HISTORICAL ARTICLE
# ============================================================

def process_historical_article(
    article
):

    timing = get_event_time(
        article,
        max_update_age_hours=
            MAX_UPDATE_AGE_HOURS
    )

    if timing is None:
        return []

    event_time = timing[
        "event_time"
    ]

    if event_time < pd.Timestamp(
        NEWS_START
    ):
        return []

    if event_time >= pd.Timestamp(
        NEWS_END
    ):
        return []

    headline = (
        article.get(
            "headline",
            ""
        )
        or ""
    )

    summary = (
        article.get(
            "summary",
            ""
        )
        or ""
    )

    symbols = clean_historical_symbols(
        article.get(
            "symbols",
            []
        )
        or []
    )

    if not symbols:
        return []

    if (
        len(symbols)
        > MAX_SYMBOLS_PER_ARTICLE
    ):
        return []

    post_move = is_post_move_article(
        headline
    )

    # Do not train on an article that says the move
    # already happened.
    if post_move:
        return []

    event_type, event_score = classify_event(
        headline,
        summary
    )

    sentiment = sentiment_score(
        headline,
        summary
    )

    future_score = future_catalyst_score(
        headline,
        summary
    )

    importance = calculate_importance(
        event_type,
        event_score,
        sentiment,
        future_score,
        False
    )

    if importance < MIN_IMPORTANCE:
        return []

    rows = []

    for symbol in symbols:

        rows.append({

            "article_id":
                article.get("id"),

            "symbol":
                symbol,

            "created_at":
                timing["created_at"],

            "updated_at":
                timing["updated_at"],

            "event_time":
                timing["event_time"],

            "update_age_hours":
                timing[
                    "update_age_hours"
                ],

            "headline":
                headline,

            "summary":
                summary,

            "event_type":
                event_type,

            "event_score":
                event_score,

            "sentiment":
                sentiment,

            "future_score":
                future_score,

            "importance":
                importance,
        })

    return rows


# ============================================================
# BUILD EVENT TABLE
# ============================================================

def build_event_table(
    articles
):

    print(
        "\n=========================================="
    )
    print(
        "          PROCESSING EVENTS"
    )
    print(
        "=========================================="
    )

    rows = []

    for article in articles:

        rows.extend(
            process_historical_article(
                article
            )
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows
    )

    for column in [
        "created_at",
        "updated_at",
        "event_time",
    ]:

        df[column] = pd.to_datetime(
            df[column],
            utc=True
        )

    df[
        "news_date"
    ] = (
        df[
            "event_time"
        ]
        .dt.tz_convert(
            NEW_YORK
        )
        .dt.date
    )

    before = len(df)

    # One training example per stock/day.
    df = (
        df
        .sort_values(
            [
                "importance",
                "event_time",
            ],
            ascending=[
                False,
                True,
            ]
        )
        .drop_duplicates(
            subset=[
                "symbol",
                "news_date",
            ],
            keep="first"
        )
        .sort_values(
            "event_time"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Raw ticker events: "
        f"{before:,}"
    )

    print(
        f"Unique ticker/day events: "
        f"{len(df):,}"
    )

    print(
        f"Unique symbols: "
        f"{df['symbol'].nunique():,}"
    )

    print(
        f"First event: "
        f"{df['event_time'].min()}"
    )

    print(
        f"Last event: "
        f"{df['event_time'].max()}"
    )

    return df


# ============================================================
# PRICE DOWNLOAD
# ============================================================

def download_one_price_batch(
    batch,
    start,
    end
):

    request = StockBarsRequest(
        symbol_or_symbols=batch,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        limit=PRICE_REQUEST_LIMIT,
        feed=DataFeed.IEX,
        adjustment=Adjustment.ALL
    )

    bars = stock_client.get_stock_bars(
        request
    )

    return (
        bars.df
        .reset_index()
    )


def download_price_history(
    symbols
):

    print(
        "\n=========================================="
    )
    print(
        "      DOWNLOADING LONG PRICE HISTORY"
    )
    print(
        "=========================================="
    )

    price_start = (
        NEWS_START
        - timedelta(
            days=PRICE_LOOKBACK_DAYS
        )
    )

    price_end = (
        NEWS_END
        + timedelta(
            days=PRICE_FORWARD_DAYS
        )
    )

    # SPY and QQQ are required for market context.
    all_symbols = set(
        symbols
    )

    all_symbols.add(
        "SPY"
    )

    all_symbols.add(
        "QQQ"
    )

    all_symbols = sorted(
        all_symbols
    )

    cache = {}

    total_batches = (
        len(all_symbols)
        + SYMBOL_BATCH_SIZE
        - 1
    ) // SYMBOL_BATCH_SIZE

    for start_index in range(
        0,
        len(all_symbols),
        SYMBOL_BATCH_SIZE
    ):

        batch = all_symbols[
            start_index:
            start_index
            + SYMBOL_BATCH_SIZE
        ]

        batch_number = (
            start_index
            // SYMBOL_BATCH_SIZE
            + 1
        )

        print(
            f"Price batch "
            f"{batch_number}/"
            f"{total_batches}"
        )

        try:

            df = download_one_price_batch(
                batch,
                price_start,
                price_end
            )

        except Exception as error:

            print(
                f"  Batch failed: "
                f"{error}"
            )

            print(
                "  Retrying individually..."
            )

            frames = []

            for symbol in batch:

                try:

                    single_df = (
                        download_one_price_batch(
                            [symbol],
                            price_start,
                            price_end
                        )
                    )

                    if not single_df.empty:
                        frames.append(
                            single_df
                        )

                except Exception:
                    pass

                time.sleep(
                    0.1
                )

            if not frames:
                continue

            df = pd.concat(
                frames,
                ignore_index=True
            )

        if df.empty:
            continue

        df[
            "timestamp"
        ] = pd.to_datetime(
            df[
                "timestamp"
            ],
            utc=True
        )

        for symbol in batch:

            symbol_df = (
                df[
                    df[
                        "symbol"
                    ]
                    == symbol
                ]
                .copy()
                .sort_values(
                    "timestamp"
                )
                .reset_index(
                    drop=True
                )
            )

            if symbol_df.empty:
                continue

            symbol_df[
                "session_date"
            ] = (
                symbol_df[
                    "timestamp"
                ]
                .dt.tz_convert(
                    NEW_YORK
                )
                .dt.date
            )

            cache[
                symbol
            ] = symbol_df

    print(
        f"\nSymbols with price data: "
        f"{len(cache):,}"
    )

    return cache


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    close,
    period=14
):

    if len(close) < (
        period + 1
    ):
        return np.nan

    delta = close.diff()

    gains = delta.clip(
        lower=0
    )

    losses = (
        -delta.clip(
            upper=0
        )
    )

    avg_gain = (
        gains
        .rolling(period)
        .mean()
        .iloc[-1]
    )

    avg_loss = (
        losses
        .rolling(period)
        .mean()
        .iloc[-1]
    )

    if pd.isna(
        avg_gain
    ) or pd.isna(
        avg_loss
    ):
        return np.nan

    if avg_loss == 0:

        if avg_gain == 0:
            return 50.0

        return 100.0

    rs = (
        avg_gain
        / avg_loss
    )

    return (
        100
        - (
            100
            / (
                1 + rs
            )
        )
    )


# ============================================================
# PRE-EVENT BAR WINDOW
# ============================================================

def get_pre_event_bars(
    price_df,
    event_time
):

    event_time = pd.to_datetime(
        event_time,
        utc=True
    )

    event_date = (
        event_time
        .tz_convert(
            NEW_YORK
        )
        .date()
    )

    # CRITICAL:
    # Strictly LESS than event date.
    #
    # We never use the event day's daily candle,
    # even if the article appeared after market close.
    pre = (
        price_df[
            price_df[
                "session_date"
            ] < event_date
        ]
        .copy()
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return pre


# ============================================================
# STRUCTURED PRICE FEATURES
# ============================================================

def calculate_price_features(
    pre,
    prefix
):

    features = {

        f"{prefix}_history_days":
            0,

        f"{prefix}_return_1d":
            np.nan,

        f"{prefix}_return_5d":
            np.nan,

        f"{prefix}_return_20d":
            np.nan,

        f"{prefix}_return_60d":
            np.nan,

        f"{prefix}_volatility_20":
            np.nan,

        f"{prefix}_volume_ratio_20":
            np.nan,

        f"{prefix}_rsi_14":
            np.nan,

        f"{prefix}_price_vs_sma20":
            np.nan,

        f"{prefix}_price_vs_sma50":
            np.nan,

        f"{prefix}_price_vs_sma200":
            np.nan,

        f"{prefix}_gap_1d":
            np.nan,

        f"{prefix}_range_1d":
            np.nan,
    }

    if pre.empty:
        return features

    close = (
        pd.to_numeric(
            pre["close"],
            errors="coerce"
        )
    )

    open_price = (
        pd.to_numeric(
            pre["open"],
            errors="coerce"
        )
    )

    high = (
        pd.to_numeric(
            pre["high"],
            errors="coerce"
        )
    )

    low = (
        pd.to_numeric(
            pre["low"],
            errors="coerce"
        )
    )

    volume = (
        pd.to_numeric(
            pre["volume"],
            errors="coerce"
        )
    )

    features[
        f"{prefix}_history_days"
    ] = len(pre)

    latest_close = float(
        close.iloc[-1]
    )

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    for days in [
        1,
        5,
        20,
        60,
    ]:

        if len(close) > days:

            old_price = float(
                close.iloc[
                    -(days + 1)
                ]
            )

            if old_price > 0:

                features[
                    f"{prefix}_return_{days}d"
                ] = (
                    latest_close
                    / old_price
                    - 1
                )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    returns = (
        close
        .pct_change()
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan
        )
        .dropna()
    )

    if len(returns) >= 20:

        features[
            f"{prefix}_volatility_20"
        ] = float(
            returns
            .tail(20)
            .std()
        )

    # --------------------------------------------------------
    # RELATIVE VOLUME
    # --------------------------------------------------------

    if len(volume) >= 21:

        previous_volume = (
            volume
            .iloc[-21:-1]
            .mean()
        )

        if (
            pd.notna(
                previous_volume
            )
            and
            previous_volume > 0
        ):

            features[
                f"{prefix}_volume_ratio_20"
            ] = (
                float(
                    volume.iloc[-1]
                )
                / float(
                    previous_volume
                )
            )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    features[
        f"{prefix}_rsi_14"
    ] = calculate_rsi(
        close
    )

    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

    for period in [
        20,
        50,
        200,
    ]:

        if len(close) >= period:

            sma = float(
                close
                .tail(period)
                .mean()
            )

            if sma > 0:

                features[
                    f"{prefix}_price_vs_sma{period}"
                ] = (
                    latest_close
                    / sma
                    - 1
                )

    # --------------------------------------------------------
    # PREVIOUS SESSION GAP
    # --------------------------------------------------------

    if len(close) >= 2:

        previous_close = float(
            close.iloc[-2]
        )

        latest_open = float(
            open_price.iloc[-1]
        )

        if previous_close > 0:

            features[
                f"{prefix}_gap_1d"
            ] = (
                latest_open
                / previous_close
                - 1
            )

    # --------------------------------------------------------
    # DAILY RANGE
    # --------------------------------------------------------

    if latest_close > 0:

        features[
            f"{prefix}_range_1d"
        ] = (
            (
                float(
                    high.iloc[-1]
                )
                - float(
                    low.iloc[-1]
                )
            )
            / latest_close
        )

    return features


# ============================================================
# STOCK FEATURES
# ============================================================

def build_stock_features(
    event,
    price_df
):

    pre = get_pre_event_bars(
        price_df,
        event[
            "event_time"
        ]
    )

    features = calculate_price_features(
        pre,
        "stock"
    )

    if pre.empty:

        features[
            "stock_last_session"
        ] = None

    else:

        features[
            "stock_last_session"
        ] = pre.iloc[-1][
            "session_date"
        ]

    return features


# ============================================================
# MARKET FEATURES
# ============================================================

def build_market_features(
    event_time,
    spy_df,
    qqq_df
):

    spy_pre = get_pre_event_bars(
        spy_df,
        event_time
    )

    qqq_pre = get_pre_event_bars(
        qqq_df,
        event_time
    )

    spy_features = calculate_price_features(
        spy_pre,
        "spy"
    )

    qqq_features = calculate_price_features(
        qqq_pre,
        "qqq"
    )

    # We don't need all SPY/QQQ features.
    wanted = {}

    for key in [
        "spy_return_5d",
        "spy_return_20d",
        "spy_volatility_20",
        "spy_price_vs_sma50",
        "spy_price_vs_sma200",

        "qqq_return_5d",
        "qqq_return_20d",
        "qqq_volatility_20",
        "qqq_price_vs_sma50",
        "qqq_price_vs_sma200",
    ]:

        if key in spy_features:

            wanted[key] = (
                spy_features[key]
            )

        elif key in qqq_features:

            wanted[key] = (
                qqq_features[key]
            )

    return wanted


# ============================================================
# RELATIVE STRENGTH FEATURES
# ============================================================

def add_relative_features(
    row
):

    pairs = [

        (
            "stock_return_5d",
            "spy_return_5d",
            "relative_spy_5d"
        ),

        (
            "stock_return_20d",
            "spy_return_20d",
            "relative_spy_20d"
        ),

        (
            "stock_return_5d",
            "qqq_return_5d",
            "relative_qqq_5d"
        ),

        (
            "stock_return_20d",
            "qqq_return_20d",
            "relative_qqq_20d"
        ),
    ]

    for stock_col, market_col, output in pairs:

        stock_value = row.get(
            stock_col
        )

        market_value = row.get(
            market_col
        )

        if (
            pd.notna(
                stock_value
            )
            and
            pd.notna(
                market_value
            )
        ):

            row[
                output
            ] = (
                stock_value
                - market_value
            )

        else:

            row[
                output
            ] = np.nan

    return row


# ============================================================
# LABEL EVENT
# ============================================================

def label_event(
    event,
    price_df
):

    event_time = pd.to_datetime(
        event[
            "event_time"
        ],
        utc=True
    )

    event_date = (
        event_time
        .tz_convert(
            NEW_YORK
        )
        .date()
    )

    future_bars = (
        price_df[
            price_df[
                "session_date"
            ] > event_date
        ]
        .copy()
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    if len(
        future_bars
    ) < SPIKE_WINDOW:

        return None

    window = (
        future_bars
        .iloc[
            :SPIKE_WINDOW
        ]
        .copy()
    )

    entry_bar = (
        window.iloc[0]
    )

    entry_price = float(
        entry_bar[
            "open"
        ]
    )

    if entry_price <= 0:
        return None

    max_price = float(
        window[
            "high"
        ].max()
    )

    min_price = float(
        window[
            "low"
        ].min()
    )

    final_close = float(
        window
        .iloc[-1][
            "close"
        ]
    )

    max_gain = (
        max_price
        / entry_price
        - 1
    )

    max_drawdown = (
        min_price
        / entry_price
        - 1
    )

    close_return = (
        final_close
        / entry_price
        - 1
    )

    return {

        "entry_date":
            entry_bar[
                "session_date"
            ],

        "entry_price":
            entry_price,

        "max_price_5d":
            max_price,

        "min_price_5d":
            min_price,

        "final_close_5d":
            final_close,

        "max_gain_5d":
            max_gain,

        "max_drawdown_5d":
            max_drawdown,

        "close_return_5d":
            close_return,

        "spike_3pct":
            int(
                max_gain
                >= SPIKE_3_PERCENT
            ),

        "spike_5pct":
            int(
                max_gain
                >= SPIKE_5_PERCENT
            ),

        "spike_10pct":
            int(
                max_gain
                >= SPIKE_10_PERCENT
            ),
    }


# ============================================================
# BUILD LABELED + FEATURE DATASET
# ============================================================

def build_labeled_dataset(
    events,
    price_cache
):

    print(
        "\n=========================================="
    )
    print(
        "      BUILDING V5 FEATURE DATASET"
    )
    print(
        "=========================================="
    )

    if (
        "SPY" not in price_cache
        or
        "QQQ" not in price_cache
    ):

        raise RuntimeError(
            "Missing SPY or QQQ market history."
        )

    spy_df = price_cache[
        "SPY"
    ]

    qqq_df = price_cache[
        "QQQ"
    ]

    labeled_rows = []

    skipped_no_price = 0
    skipped_no_future = 0

    # Market features only depend on calendar date
    # because we use the completed session before
    # the news date.
    market_cache = {}

    for _, event in (
        events.iterrows()
    ):

        symbol = event[
            "symbol"
        ]

        if symbol not in price_cache:

            skipped_no_price += 1
            continue

        label = label_event(
            event,
            price_cache[
                symbol
            ]
        )

        if label is None:

            skipped_no_future += 1
            continue

        row = event.to_dict()

        stock_features = (
            build_stock_features(
                event,
                price_cache[
                    symbol
                ]
            )
        )

        row.update(
            stock_features
        )

        event_date = (
            pd.to_datetime(
                event[
                    "event_time"
                ],
                utc=True
            )
            .tz_convert(
                NEW_YORK
            )
            .date()
        )

        if event_date not in market_cache:

            market_cache[
                event_date
            ] = (
                build_market_features(
                    event[
                        "event_time"
                    ],
                    spy_df,
                    qqq_df
                )
            )

        row.update(
            market_cache[
                event_date
            ]
        )

        row = add_relative_features(
            row
        )

        row.update(
            label
        )

        labeled_rows.append(
            row
        )

        if (
            len(labeled_rows)
            % 1000
            == 0
        ):

            print(
                f"Labeled + featured "
                f"{len(labeled_rows):,} events..."
            )

    print(
        f"\nSkipped - no prices: "
        f"{skipped_no_price:,}"
    )

    print(
        f"Skipped - insufficient future data: "
        f"{skipped_no_future:,}"
    )

    return pd.DataFrame(
        labeled_rows
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    dataset
):

    print(
        "\n=========================================="
    )
    print(
        "          V5 DATASET SUMMARY"
    )
    print(
        "=========================================="
    )

    total = len(
        dataset
    )

    print(
        f"\nTotal examples: "
        f"{total:,}"
    )

    print(
        f"Unique stocks: "
        f"{dataset['symbol'].nunique():,}"
    )

    print(
        f"First event: "
        f"{dataset['event_time'].min()}"
    )

    print(
        f"Last event:  "
        f"{dataset['event_time'].max()}"
    )

    for column, label in [

        (
            "spike_3pct",
            "3%"
        ),

        (
            "spike_5pct",
            "5%"
        ),

        (
            "spike_10pct",
            "10%"
        ),
    ]:

        count = int(
            dataset[
                column
            ].sum()
        )

        print(
            f"\n{label} spikes:"
        )

        print(
            f"  {count:,} "
            f"({count / total:.2%})"
        )

    print(
        "\nAverage maximum 5-day gain:"
    )

    print(
        f"  "
        f"{dataset['max_gain_5d'].mean():.2%}"
    )

    print(
        "\nAverage 5-day close return:"
    )

    print(
        f"  "
        f"{dataset['close_return_5d'].mean():.2%}"
    )

    if (
        "stock_history_days"
        in dataset.columns
    ):

        history_200 = (
            dataset[
                "stock_history_days"
            ] >= 200
        ).mean()

        print(
            "\nEvents with 200+ prior sessions:"
        )

        print(
            f"  {history_200:.2%}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n=========================================="
    )
    print(
        "        V5 SPIKE DATASET BUILDER"
    )
    print(
        "=========================================="
    )

    print(
        f"\nNews period:"
    )

    print(
        f"{NEWS_START}"
    )

    print(
        "→"
    )

    print(
        f"{NEWS_END}"
    )

    print(
        f"\nPrice lookback: "
        f"{PRICE_LOOKBACK_DAYS} days"
    )

    articles = fetch_historical_news()

    events = build_event_table(
        articles
    )

    if events.empty:

        print(
            "\nNo usable events."
        )

        raise SystemExit

    symbols = (
        events[
            "symbol"
        ]
        .unique()
        .tolist()
    )

    price_cache = (
        download_price_history(
            symbols
        )
    )

    dataset = (
        build_labeled_dataset(
            events,
            price_cache
        )
    )

    if dataset.empty:

        print(
            "\nNo labeled rows produced."
        )

        raise SystemExit

    dataset = (
        dataset
        .sort_values(
            "event_time"
        )
        .reset_index(
            drop=True
        )
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print_summary(
        dataset
    )

    print(
        "\nDataset saved to:"
    )

    print(
        OUTPUT_FILE
    )