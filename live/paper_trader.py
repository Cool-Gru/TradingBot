import os
import time
from datetime import datetime, timezone

import joblib
import pandas as pd

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from TradingBot.archive.data import get_daily_data
from ml.features import build_features
from ml.market_features import build_market_features

from live.risk_manager import can_open_trade
from live.position_manager import (
    record_entry,
    check_position_exit
)


# ============================================================
# SETTINGS
# ============================================================

# KEEP THIS TRUE FOR TESTING. SET TO FALSE TO ENABLE LIVE PAPER TRADING.
DRY_RUN = False

MODEL_PATH = "ml/trading_model_v3.joblib"

DATA_START = datetime(
    2025,
    1,
    1,
    tzinfo=timezone.utc
)

DATA_END = datetime.now(
    timezone.utc
)

# How long to wait for a PAPER market order
# to report a fill.
FILL_TIMEOUT_SECONDS = 30

FILL_CHECK_INTERVAL = 1


# ============================================================
# ALPACA CONNECTION
# ============================================================

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")


if not API_KEY or not SECRET_KEY:
    raise RuntimeError(
        "Missing Alpaca API keys in .env"
    )


trading_client = TradingClient(
    API_KEY,
    SECRET_KEY,
    paper=True
)


# ============================================================
# CHECK CURRENT POSITION / EXIT FIRST
# ============================================================

exit_check = check_position_exit()


if exit_check["action"] != "NONE":

    print("\n==========================================")
    print("          POSITION STATUS")
    print("==========================================")

    print(
        exit_check["message"]
    )


    # If a position exists or an exit is due,
    # do NOT scan for another entry.
    if exit_check["action"] in [
        "HOLD",
        "DRY_RUN_SELL",
        "SELL",
        "WAIT",
        "WARNING"
    ]:

        raise SystemExit


# ============================================================
# LOAD MODEL
# ============================================================

print("\n==========================================")
print("             PAPER TRADER")
print("==========================================")

print("\nLoading V3 model...")


bundle = joblib.load(
    MODEL_PATH
)


model = bundle[
    "model"
]

FEATURES = bundle[
    "features"
]

SYMBOLS = bundle[
    "symbols"
]

BUY_THRESHOLD = bundle[
    "buy_threshold"
]


print(
    "Model loaded."
)

print(
    f"Threshold: "
    f"{BUY_THRESHOLD:.0%}"
)

print(
    f"Symbols: "
    f"{', '.join(SYMBOLS)}"
)


# ============================================================
# MARKET CLOCK
# ============================================================

clock = trading_client.get_clock()


print(
    f"\nMarket open: "
    f"{clock.is_open}"
)

print(
    f"Current market time: "
    f"{clock.timestamp}"
)


# ============================================================
# DOWNLOAD MARKET CONTEXT
# ============================================================

print(
    "\nDownloading market context..."
)


spy_raw = get_daily_data(
    "SPY",
    DATA_START,
    DATA_END
)


qqq_raw = get_daily_data(
    "QQQ",
    DATA_START,
    DATA_END
)


# ============================================================
# SCORE STOCKS
# ============================================================

candidates = []


for symbol in SYMBOLS:

    print(
        f"Analyzing {symbol}..."
    )


    raw = get_daily_data(
        symbol,
        DATA_START,
        DATA_END
    )


    stock_features = build_features(
        raw
    )


    combined = build_market_features(
        stock_features,
        spy_raw,
        qqq_raw
    )


    if len(combined) == 0:

        print(
            f"  Skipping {symbol}: "
            f"not enough data."
        )

        continue


    latest = combined.iloc[-1]


    X = pd.DataFrame(
        [
            latest[
                FEATURES
            ].to_dict()
        ]
    )


    probability = (
        model
        .predict_proba(
            X
        )[0][1]
    )


    bull_market = (
        latest[
            "spy_vs_sma200"
        ] > 0

        and

        latest[
            "qqq_vs_sma200"
        ] > 0
    )


    print(
        f"  AI probability: "
        f"{probability:.2%}"
    )

    print(
        f"  Bull regime: "
        f"{bull_market}"
    )


    candidates.append({

        "symbol":
            symbol,

        "probability":
            probability,

        "bull_market":
            bull_market,

        "close":
            float(
                latest["close"]
            ),

        "timestamp":
            latest["timestamp"]
    })


# ============================================================
# VALID CANDIDATES
# ============================================================

valid_candidates = [

    candidate

    for candidate in candidates

    if (
        candidate[
            "probability"
        ] >= BUY_THRESHOLD

        and

        candidate[
            "bull_market"
        ]
    )
]


print("\n==========================================")
print("             AI DECISION")
print("==========================================")


if len(valid_candidates) == 0:

    print(
        "\nNo qualifying trades today."
    )

    raise SystemExit


# ============================================================
# PICK BEST CANDIDATE
# ============================================================

valid_candidates.sort(

    key=lambda x:
        x[
            "probability"
        ],

    reverse=True
)


best = valid_candidates[0]

symbol = best[
    "symbol"
]

probability = best[
    "probability"
]


print(
    f"\nBest candidate: "
    f"{symbol}"
)

print(
    f"AI confidence: "
    f"{probability:.2%}"
)

print(
    f"Latest close: "
    f"${best['close']:.2f}"
)

print(
    f"Signal timestamp: "
    f"{best['timestamp']}"
)


# ============================================================
# RISK MANAGER
# ============================================================

risk = can_open_trade(
    symbol
)


if not risk[
    "approved"
]:

    print("\n==========================================")
    print("      TRADE REJECTED BY RISK MANAGER")
    print("==========================================")

    for reason in risk[
        "reasons"
    ]:

        print(
            f" - {reason}"
        )

    raise SystemExit


budget = risk[
    "budget"
]


print(
    f"\nApproved budget: "
    f"${budget:,.2f}"
)


# ============================================================
# POSITION SIZE
# ============================================================

estimated_price = best[
    "close"
]


shares = int(
    budget
    // estimated_price
)


if shares <= 0:

    print(
        "\nNot enough budget "
        "to purchase one share."
    )

    raise SystemExit


estimated_value = (
    shares
    * estimated_price
)


print(
    f"Shares: "
    f"{shares}"
)

print(
    f"Estimated position value: "
    f"${estimated_value:,.2f}"
)


# ============================================================
# ORDER REQUEST
# ============================================================

order_request = MarketOrderRequest(
    symbol=symbol,
    qty=shares,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY
)


# ============================================================
# DRY RUN
# ============================================================

if DRY_RUN:

    print("\n==========================================")
    print("               DRY RUN")
    print("==========================================")

    print(
        "\nNO ORDER WAS SUBMITTED."
    )

    print(
        f"\nWould BUY "
        f"{shares} shares of {symbol}"
    )

    print(
        f"AI confidence: "
        f"{probability:.2%}"
    )

    print(
        f"Estimated price: "
        f"${estimated_price:.2f}"
    )

    print(
        f"Estimated value: "
        f"${estimated_value:,.2f}"
    )

    print(
        f"Maximum approved budget: "
        f"${budget:,.2f}"
    )

    raise SystemExit


# ============================================================
# MARKET OPEN SAFETY
# ============================================================

# Refresh the clock immediately before submitting.
clock = trading_client.get_clock()


if not clock.is_open:

    print(
        "\nMarket is closed."
    )

    print(
        "No PAPER order submitted."
    )

    raise SystemExit


# ============================================================
# SUBMIT PAPER BUY
# ============================================================

print(
    "\nSubmitting PAPER BUY order..."
)


order = trading_client.submit_order(
    order_data=order_request
)


print(
    f"Order submitted: "
    f"{order.id}"
)


# ============================================================
# WAIT FOR PAPER FILL
# ============================================================

print(
    "Waiting for simulated fill..."
)


filled_order = None

start_time = time.time()


while (
    time.time() - start_time
    < FILL_TIMEOUT_SECONDS
):

    current_order = (
        trading_client
        .get_order_by_id(
            order.id
        )
    )


    if (
        current_order.filled_avg_price
        is not None
    ):

        filled_order = (
            current_order
        )

        break


    time.sleep(
        FILL_CHECK_INTERVAL
    )


# ============================================================
# FILL FAILED / TIMED OUT
# ============================================================

if filled_order is None:

    print("\n==========================================")
    print("           FILL NOT CONFIRMED")
    print("==========================================")

    print(
        "\nThe order was submitted, "
        "but a fill was not confirmed "
        "within the timeout."
    )

    print(
        f"Order ID: "
        f"{order.id}"
    )

    print(
        "\nNo entry was written to "
        "trade_log.csv yet."
    )

    raise SystemExit


# ============================================================
# ACTUAL PAPER FILL
# ============================================================

fill_price = float(
    filled_order.filled_avg_price
)


filled_qty = float(
    filled_order.filled_qty
)


print("\n==========================================")
print("             PAPER BUY FILLED")
print("==========================================")

print(
    f"\nSymbol: "
    f"{symbol}"
)

print(
    f"Filled quantity: "
    f"{filled_qty}"
)

print(
    f"Actual fill price: "
    f"${fill_price:.2f}"
)

print(
    f"Order status: "
    f"{filled_order.status}"
)

print(
    f"Order ID: "
    f"{filled_order.id}"
)


# ============================================================
# RECORD ACTUAL FILL
# ============================================================

record_entry(
    symbol=symbol,
    entry_price=fill_price,
    quantity=filled_qty,
    probability=probability
)


print(
    "\nTrade recorded in:"
)

print(
    "live/trade_log.csv"
)

print(
    "\nThis was PAPER TRADING only."
)