import os
import json
import time

from datetime import (
    datetime,
    timedelta,
    timezone,
)

from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from alpaca.trading.client import (
    TradingClient
)

from alpaca.trading.requests import (
    MarketOrderRequest,
    GetCalendarRequest,
)

from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
)

from alpaca.data.historical import (
    StockHistoricalDataClient
)

from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestTradeRequest,
)

from alpaca.data.timeframe import (
    TimeFrame
)

from alpaca.data.enums import (
    DataFeed,
    Adjustment,
)

from news.news_scanner import (
    scan_news
)

from news.build_spike_dataset import (
    build_stock_features,
    build_market_features,
    add_relative_features,
    get_pre_event_bars,
)

from live.risk_manager import (
    can_open_trade
)

from live.position_manager import (
    record_entry,
    check_position_exit,
)


# ============================================================
# SETTINGS
# ============================================================

DRY_RUN = True

MODEL_PATH = (
    "news/spike_model_v5.joblib"
)

PENDING_SIGNAL_PATH = (
    "live/v5_pending_signal.json"
)


# V5 OOS results:
#
# >= 65%  -> 53.45% precision
# >= 70%  -> 54.99%
# >= 75%  -> 58.61%
#
# Start paper testing at 70%.
SPIKE_THRESHOLD = 0.70


# Only score the strongest live news articles.
MAX_CANDIDATES_TO_SCORE = 25


PRICE_LOOKBACK_DAYS = 450


# Even though the general risk manager allows
# 20%, news trades start at only 5% of equity.
NEWS_POSITION_PERCENT = 0.05


MIN_SHARE_PRICE = 5.00


# We validated entry at the next market open.
#
# If the script isn't run until hours later,
# do NOT pretend that is the same strategy.
ENTRY_WINDOW_MINUTES = 30


FILL_TIMEOUT_SECONDS = 30

FILL_CHECK_INTERVAL = 1


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


trading_client = TradingClient(
    API_KEY,
    SECRET_KEY,
    paper=True
)


stock_client = (
    StockHistoricalDataClient(
        API_KEY,
        SECRET_KEY
    )
)


# ============================================================
# LOAD V5 MODEL
# ============================================================

if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(

        f"Could not find {MODEL_PATH}.\n"
        "Make sure the V5 trainer finished and "
        "printed 'V5 MODEL SAVED'."
    )


bundle = joblib.load(
    MODEL_PATH
)


model = bundle[
    "model"
]


FEATURE_COLUMNS = bundle[
    "features"
]


NUMERIC_COLUMNS = bundle[
    "numeric_features"
]


# ============================================================
# PENDING SIGNAL STORAGE
# ============================================================

def load_pending_signal():

    if not os.path.exists(
        PENDING_SIGNAL_PATH
    ):

        return None


    with open(
        PENDING_SIGNAL_PATH,
        "r"
    ) as file:

        return json.load(
            file
        )


def save_pending_signal(
    signal
):

    with open(
        PENDING_SIGNAL_PATH,
        "w"
    ) as file:

        json.dump(
            signal,
            file,
            indent=4,
            default=str
        )


def clear_pending_signal():

    if os.path.exists(
        PENDING_SIGNAL_PATH
    ):

        os.remove(
            PENDING_SIGNAL_PATH
        )


# ============================================================
# PRICE HISTORY
# ============================================================

def download_history(
    symbol
):

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

        symbol_or_symbols=
            symbol,

        timeframe=
            TimeFrame.Day,

        start=
            start,

        end=
            end,

        feed=
            DataFeed.IEX,

        adjustment=
            Adjustment.ALL
    )


    bars = (
        stock_client
        .get_stock_bars(
            request
        )
    )


    df = (
        bars.df
        .reset_index()
    )


    if df.empty:

        return None


    if "symbol" in df.columns:

        df = (
            df[
                df[
                    "symbol"
                ]
                == symbol
            ]
            .copy()
        )


    if df.empty:

        return None


    df[
        "timestamp"
    ] = pd.to_datetime(
        df[
            "timestamp"
        ],
        utc=True
    )


    df[
        "session_date"
    ] = (
        df[
            "timestamp"
        ]
        .dt.tz_convert(
            NEW_YORK
        )
        .dt.date
    )


    return (
        df
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# CURRENT PRICE
# ============================================================

def get_latest_trade_price(
    symbol
):

    request = StockLatestTradeRequest(

        symbol_or_symbols=
            symbol,

        feed=
            DataFeed.IEX
    )


    trades = (
        stock_client
        .get_stock_latest_trade(
            request
        )
    )


    if symbol not in trades:

        return None


    price = float(
        trades[
            symbol
        ].price
    )


    return price


# ============================================================
# FIND NEXT MARKET SESSION AFTER NEWS DATE
# ============================================================

def get_entry_open(
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


    search_start = (
        event_date
        + timedelta(
            days=1
        )
    )


    search_end = (
        event_date
        + timedelta(
            days=10
        )
    )


    request = GetCalendarRequest(

        start=
            search_start,

        end=
            search_end
    )


    calendar = (
        trading_client
        .get_calendar(
            filters=request
        )
    )


    if not calendar:

        return None


    first_session = (
        calendar[0]
    )


    open_time = (
        first_session.open
    )


    if open_time.tzinfo is None:

        open_time = (
            open_time
            .replace(
                tzinfo=NEW_YORK
            )
        )

    else:

        open_time = (
            open_time
            .astimezone(
                NEW_YORK
            )
        )


    return open_time


# ============================================================
# EXPAND NEWS INTO ONE EVENT PER TICKER
# ============================================================

def build_live_candidates(
    news_results
):

    best = {}


    for article in news_results:

        if article.get(
            "post_move",
            False
        ):

            continue


        symbols = article.get(
            "symbols",
            []
        )


        event_time = pd.to_datetime(

            article[
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


        for symbol in symbols:

            candidate = dict(
                article
            )


            candidate[
                "symbol"
            ] = symbol


            key = (
                symbol,
                event_date
            )


            current = best.get(
                key
            )


            if current is None:

                best[
                    key
                ] = candidate

                continue


            if (
                candidate[
                    "importance"
                ]
                >
                current[
                    "importance"
                ]
            ):

                best[
                    key
                ] = candidate


    candidates = list(
        best.values()
    )


    candidates.sort(

        key=lambda x:
            x[
                "importance"
            ],

        reverse=True
    )


    return candidates[
        :MAX_CANDIDATES_TO_SCORE
    ]


# ============================================================
# SCORE ONE EVENT
# ============================================================

def score_event(
    event,
    spy_df,
    qqq_df
):

    symbol = event[
        "symbol"
    ]


    # --------------------------------------------------------
    # Confirm Alpaca considers the equity tradable
    # --------------------------------------------------------

    try:

        asset = (
            trading_client
            .get_asset(
                symbol
            )
        )

    except Exception:

        return None


    if not asset.tradable:

        return None


    # --------------------------------------------------------
    # STOCK HISTORY
    # --------------------------------------------------------

    stock_df = (
        download_history(
            symbol
        )
    )


    if stock_df is None:

        return None


    pre = get_pre_event_bars(

        stock_df,

        event[
            "event_time"
        ]
    )


    if len(
        pre
    ) < 20:

        return None


    previous_close = float(
        pre
        .iloc[-1][
            "close"
        ]
    )


    if (
        previous_close
        < MIN_SHARE_PRICE
    ):

        return None


    # --------------------------------------------------------
    # BUILD EXACT V5 FEATURES
    # --------------------------------------------------------

    row = {

        "text":
            (
                str(
                    event.get(
                        "headline",
                        ""
                    )
                )
                + " "
                + str(
                    event.get(
                        "summary",
                        ""
                    )
                )
            ),

        "event_type":
            event.get(
                "event_type",
                "OTHER"
            ),

        "event_score":
            event.get(
                "event_score",
                0
            ),

        "sentiment":
            event.get(
                "sentiment",
                0
            ),

        "future_score":
            event.get(
                "future_score",
                0
            ),

        "importance":
            event.get(
                "importance",
                0
            ),
    }


    stock_features = (
        build_stock_features(
            event,
            stock_df
        )
    )


    row.update(
        stock_features
    )


    market_features = (
        build_market_features(

            event[
                "event_time"
            ],

            spy_df,

            qqq_df
        )
    )


    row.update(
        market_features
    )


    row = (
        add_relative_features(
            row
        )
    )


    # --------------------------------------------------------
    # ENSURE EVERY TRAINING FEATURE EXISTS
    # --------------------------------------------------------

    for column in (
        NUMERIC_COLUMNS
    ):

        if column not in row:

            row[
                column
            ] = np.nan


    X = pd.DataFrame(
        [
            {
                column:
                    row.get(
                        column,
                        np.nan
                    )

                for column in
                FEATURE_COLUMNS
            }
        ]
    )


    probability = float(

        model
        .predict_proba(
            X
        )[0][1]
    )


    entry_open = (
        get_entry_open(
            event[
                "event_time"
            ]
        )
    )


    if entry_open is None:

        return None


    return {

        "symbol":
            symbol,

        "score":
            probability,

        "headline":
            event.get(
                "headline",
                ""
            ),

        "event_type":
            event.get(
                "event_type",
                "OTHER"
            ),

        "sentiment":
            event.get(
                "sentiment",
                0
            ),

        "importance":
            event.get(
                "importance",
                0
            ),

        "event_time":
            str(
                event[
                    "event_time"
                ]
            ),

        "intended_entry":
            entry_open.isoformat(),

        "previous_close":
            previous_close,

        "history_days":
            int(
                stock_features.get(
                    "stock_history_days",
                    0
                )
            ),
    }


# ============================================================
# SCAN AND CREATE PENDING SIGNAL
# ============================================================

def scan_and_queue():

    print(
        "\n=========================================="
    )

    print(
        "          V5 LIVE NEWS SCANNER"
    )

    print(
        "=========================================="
    )


    print(
        f"\nSpike threshold: "
        f"{SPIKE_THRESHOLD:.0%}"
    )


    news_results = (
        scan_news()
    )


    candidates = (
        build_live_candidates(
            news_results
        )
    )


    print(
        f"\nCandidate ticker-events: "
        f"{len(candidates)}"
    )


    if not candidates:

        print(
            "\nNo candidates."
        )

        return


    print(
        "\nDownloading SPY / QQQ context..."
    )


    spy_df = (
        download_history(
            "SPY"
        )
    )


    qqq_df = (
        download_history(
            "QQQ"
        )
    )


    if (
        spy_df is None
        or
        qqq_df is None
    ):

        raise RuntimeError(
            "Could not download SPY/QQQ data."
        )


    clock = (
        trading_client
        .get_clock()
    )


    now = (
        clock.timestamp
    )


    scored = []


    for candidate in candidates:

        symbol = candidate[
            "symbol"
        ]


        print(
            f"\nScoring {symbol}..."
        )


        result = score_event(

            candidate,

            spy_df,

            qqq_df
        )


        if result is None:

            print(
                "  Skipped."
            )

            continue


        intended_entry = (
            pd.to_datetime(
                result[
                    "intended_entry"
                ]
            )
        )


        # If the proper next-session entry has
        # already happened, this news is stale.
        if intended_entry <= now:

            print(
                "  Stale for V5 entry."
            )

            continue


        print(
            f"  V5 score: "
            f"{result['score']:.2%}"
        )


        print(
            f"  Event: "
            f"{result['event_type']}"
        )


        print(
            f"  Intended entry: "
            f"{result['intended_entry']}"
        )


        scored.append(
            result
        )


    if not scored:

        print(
            "\nNo non-stale scored candidates."
        )

        return


    scored.sort(

        key=lambda x:
            x[
                "score"
            ],

        reverse=True
    )


    print(
        "\n=========================================="
    )

    print(
        "             TOP V5 SIGNALS"
    )

    print(
        "=========================================="
    )


    for result in scored[:10]:

        print(
            f"\n"
            f"{result['symbol']:5}  "
            f"{result['score']:.2%}  "
            f"{result['event_type']}"
        )


        print(
            f"  "
            f"{result['headline']}"
        )


    best = (
        scored[0]
    )


    if (
        best[
            "score"
        ]
        < SPIKE_THRESHOLD
    ):

        print(
            "\nNo signal exceeded "
            f"{SPIKE_THRESHOLD:.0%}."
        )

        return


    best[
        "queued_at"
    ] = datetime.now(
        timezone.utc
    ).isoformat()


    save_pending_signal(
        best
    )


    print(
        "\n=========================================="
    )

    print(
        "           SIGNAL QUEUED"
    )

    print(
        "=========================================="
    )


    print(
        f"\nSymbol: "
        f"{best['symbol']}"
    )


    print(
        f"V5 score: "
        f"{best['score']:.2%}"
    )


    print(
        f"Entry time: "
        f"{best['intended_entry']}"
    )


    print(
        "\nThis signal has NOT been traded."
    )


# ============================================================
# EXECUTE PENDING SIGNAL
# ============================================================

def execute_pending(
    pending
):

    symbol = (
        pending[
            "symbol"
        ]
    )


    intended_entry = (
        pd.to_datetime(
            pending[
                "intended_entry"
            ]
        )
    )


    clock = (
        trading_client
        .get_clock()
    )


    now = (
        clock.timestamp
    )


    latest_allowed = (

        intended_entry

        + pd.Timedelta(
            minutes=
                ENTRY_WINDOW_MINUTES
        )
    )


    print(
        "\n=========================================="
    )

    print(
        "          PENDING V5 SIGNAL"
    )

    print(
        "=========================================="
    )


    print(
        f"\nSymbol: "
        f"{symbol}"
    )


    print(
        f"Score: "
        f"{pending['score']:.2%}"
    )


    print(
        f"Intended entry: "
        f"{intended_entry}"
    )


    # --------------------------------------------------------
    # TOO EARLY
    # --------------------------------------------------------

    if now < intended_entry:

        print(
            "\nWaiting for intended "
            "market session."
        )

        return


    # --------------------------------------------------------
    # TOO LATE
    # --------------------------------------------------------

    if now > latest_allowed:

        print(
            "\nSignal expired."
        )

        print(
            "Entry window was missed."
        )


        clear_pending_signal()

        return


    # --------------------------------------------------------
    # MARKET MUST BE OPEN
    # --------------------------------------------------------

    if not clock.is_open:

        print(
            "\nMarket is not open yet."
        )

        return


    # --------------------------------------------------------
    # RISK MANAGER
    # --------------------------------------------------------

    risk = (
        can_open_trade(
            symbol
        )
    )


    if not risk[
        "approved"
    ]:

        print(
            "\nTRADE REJECTED:"
        )


        for reason in risk[
            "reasons"
        ]:

            print(
                f" - {reason}"
            )


        return


    # --------------------------------------------------------
    # CURRENT PRICE
    # --------------------------------------------------------

    current_price = (
        get_latest_trade_price(
            symbol
        )
    )


    if (
        current_price is None
        or
        current_price <= 0
    ):

        print(
            "\nCould not obtain current price."
        )

        return


    if current_price < MIN_SHARE_PRICE:

        print(
            "\nPrice fell below minimum."
        )


        clear_pending_signal()

        return


    # --------------------------------------------------------
    # SMALLER V5 PAPER-TRADING BUDGET
    # --------------------------------------------------------

    equity = (
        risk[
            "account"
        ][
            "equity"
        ]
    )


    cash = (
        risk[
            "account"
        ][
            "cash"
        ]
    )


    v5_limit = (
        equity
        * NEWS_POSITION_PERCENT
    )


    budget = min(

        risk[
            "budget"
        ],

        v5_limit,

        cash
    )


    # Keep a tiny sizing buffer.
    usable_budget = (
        budget
        * 0.98
    )


    shares = int(

        usable_budget
        // current_price
    )


    if shares <= 0:

        print(
            "\nNot enough budget for "
            "one share."
        )

        return


    estimated_value = (
        shares
        * current_price
    )


    print(
        f"\nCurrent price: "
        f"${current_price:,.2f}"
    )


    print(
        f"V5 trade budget: "
        f"${budget:,.2f}"
    )


    print(
        f"Shares: "
        f"{shares}"
    )


    print(
        f"Estimated value: "
        f"${estimated_value:,.2f}"
    )


    # --------------------------------------------------------
    # DRY RUN
    # --------------------------------------------------------

    if DRY_RUN:

        print(
            "\n=========================================="
        )

        print(
            "                DRY RUN"
        )

        print(
            "=========================================="
        )


        print(
            f"\nWould BUY "
            f"{shares} shares of "
            f"{symbol}."
        )


        print(
            f"V5 spike score: "
            f"{pending['score']:.2%}"
        )


        print(
            "\nNO ORDER WAS SUBMITTED."
        )

        return


    # --------------------------------------------------------
    # PAPER ORDER
    # --------------------------------------------------------

    order_request = MarketOrderRequest(

        symbol=
            symbol,

        qty=
            shares,

        side=
            OrderSide.BUY,

        time_in_force=
            TimeInForce.DAY
    )


    print(
        "\nSubmitting PAPER V5 order..."
    )


    order = (
        trading_client
        .submit_order(
            order_data=
                order_request
        )
    )


    print(
        f"Order ID: "
        f"{order.id}"
    )


    # --------------------------------------------------------
    # WAIT FOR FILL
    # --------------------------------------------------------

    start_time = (
        time.time()
    )


    filled_order = None


    while (
        time.time()
        - start_time
        <
        FILL_TIMEOUT_SECONDS
    ):

        current_order = (
            trading_client
            .get_order_by_id(
                order.id
            )
        )


        if (
            current_order
            .filled_avg_price
            is not None
        ):

            filled_order = (
                current_order
            )

            break


        time.sleep(
            FILL_CHECK_INTERVAL
        )


    if filled_order is None:

        print(
            "\nOrder submitted, but "
            "fill was not confirmed yet."
        )


        print(
            "Pending signal was NOT "
            "submitted again."
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

        symbol=
            symbol,

        entry_price=
            fill_price,

        quantity=
            filled_qty,

        probability=
            pending[
                "score"
            ]
    )


    clear_pending_signal()


    print(
        "\n=========================================="
    )

    print(
        "           V5 PAPER BUY FILLED"
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
        f"${fill_price:,.2f}"
    )


    print(
        f"V5 score: "
        f"{pending['score']:.2%}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n=========================================="
    )

    print(
        "          V5 NEWS PAPER TRADER"
    )

    print(
        "=========================================="
    )


    print(
        f"\nDRY RUN: "
        f"{DRY_RUN}"
    )


    print(
        f"V5 threshold: "
        f"{SPIKE_THRESHOLD:.0%}"
    )


    print(
        "Alpaca mode: PAPER"
    )


    # --------------------------------------------------------
    # HANDLE EXISTING POSITION FIRST
    # --------------------------------------------------------

    exit_check = (
        check_position_exit()
    )


    if exit_check[
        "action"
    ] != "NONE":

        print(
            "\nPosition manager:"
        )


        print(
            exit_check[
                "message"
            ]
        )


        raise SystemExit


    # --------------------------------------------------------
    # EXISTING PENDING SIGNAL
    # --------------------------------------------------------

    pending = (
        load_pending_signal()
    )


    if pending is not None:

        execute_pending(
            pending
        )

        raise SystemExit


    # --------------------------------------------------------
    # NO PENDING SIGNAL
    # --------------------------------------------------------

    clock = (
        trading_client
        .get_clock()
    )


    print(
        f"\nMarket open: "
        f"{clock.is_open}"
    )


    print(
        f"Market time: "
        f"{clock.timestamp}"
    )


    # --------------------------------------------------------
    # DURING MARKET HOURS:
    #
    # Do not discover a fresh article and enter
    # immediately. That is NOT what V5 was trained on.
    # --------------------------------------------------------

    if clock.is_open:

        print(
            "\nNo pending V5 signal."
        )


        print(
            "Not opening a same-session "
            "news trade."
        )


        print(
            "Run the scanner after market "
            "close to create tomorrow's signal."
        )


        raise SystemExit


    # --------------------------------------------------------
    # MARKET CLOSED:
    # SCORE NEWS AND QUEUE NEXT-SESSION SIGNAL
    # --------------------------------------------------------

    scan_and_queue()