from datetime import datetime, timezone

from TradingBot.archive.data import get_daily_data
from strategies.sma_cross import add_signals
from TradingBot.archive.backtester import run_backtest
from TradingBot.archive.metrics import calculate_metrics


# ==========================================
# SETTINGS
# ==========================================

SYMBOL = "AAPL"
STARTING_CASH = 10_000

# Winner from our 2021-2024 optimization
FAST = 30
SLOW = 50
TREND = 100


# ==========================================
# DATA / TEST PERIODS
# ==========================================

# Extra historical data used ONLY to calculate
# moving averages before the test begins.
DATA_START = datetime(
    2024, 6, 1,
    tzinfo=timezone.utc
)

# Actual unseen test begins here.
TEST_START = datetime(
    2025, 1, 1,
    tzinfo=timezone.utc
)

TEST_END = datetime.now(
    timezone.utc
)


# ==========================================
# DOWNLOAD DATA
# ==========================================

print("Downloading data...")

df = get_daily_data(
    SYMBOL,
    DATA_START,
    TEST_END
)

print(
    f"Downloaded {len(df)} trading days "
    f"(including warm-up data)."
)


# ==========================================
# CALCULATE INDICATORS
# ==========================================

# IMPORTANT:
# Calculate indicators BEFORE removing the
# warm-up period.
#
# This allows SMA 30, SMA 50, and SMA 100
# to already exist when 2025 begins.

df = add_signals(
    df,
    fast=FAST,
    slow=SLOW,
    trend_filter=TREND
)


# ==========================================
# REMOVE WARM-UP PERIOD
# ==========================================

# Everything before 2025 was ONLY used
# for calculating indicators.
#
# No trades before TEST_START are allowed.

df = df[
    df["timestamp"] >= TEST_START
].copy()

df = df.reset_index(drop=True)

print(
    f"Testing on {len(df)} trading days "
    f"from {TEST_START.date()} onward."
)


# ==========================================
# RUN BACKTEST
# ==========================================

equity, trades = run_backtest(
    df,
    starting_cash=STARTING_CASH
)

results = calculate_metrics(
    equity,
    trades,
    STARTING_CASH
)


# ==========================================
# BUY & HOLD BENCHMARK
# ==========================================

first_price = df.iloc[0]["open"]
last_price = df.iloc[-1]["close"]

buy_hold_shares = int(
    STARTING_CASH // first_price
)

buy_hold_cash = (
    STARTING_CASH
    - buy_hold_shares * first_price
)

buy_hold_value = (
    buy_hold_cash
    + buy_hold_shares * last_price
)

buy_hold_return = (
    buy_hold_value / STARTING_CASH - 1
) * 100


# ==========================================
# RESULTS
# ==========================================

print("\n===================================")
print("       OUT-OF-SAMPLE TEST")
print("===================================")

print(f"\nSymbol:       {SYMBOL}")

print(
    f"Strategy:     "
    f"SMA {FAST}/{SLOW} "
    f"Trend {TREND}"
)

print(
    f"Test Period:  "
    f"{TEST_START.date()} "
    f"to {TEST_END.date()}"
)

print("\n--- BOT ---")

print(
    f"Ending Value: "
    f"${results['final_value']:,.2f}"
)

print(
    f"Return:       "
    f"{results['total_return']:.2f}%"
)

print(
    f"Max Drawdown: "
    f"{results['max_drawdown']:.2f}%"
)

print(
    f"Win Rate:     "
    f"{results['win_rate']:.2f}%"
)

print(
    f"Avg Trade:    "
    f"{results['avg_trade']:.2f}%"
)

print(
    f"Best Trade:   "
    f"{results['best_trade']:.2f}%"
)

print(
    f"Worst Trade:  "
    f"{results['worst_trade']:.2f}%"
)

print(
    f"Trades:       "
    f"{results['trades']}"
)


print("\n--- BUY & HOLD ---")

print(
    f"Ending Value: "
    f"${buy_hold_value:,.2f}"
)

print(
    f"Return:       "
    f"{buy_hold_return:.2f}%"
)


# ==========================================
# INDIVIDUAL TRADES
# ==========================================

if len(trades) > 0:

    print("\n===================================")
    print("              TRADES")
    print("===================================")

    for number, trade in enumerate(
        trades,
        start=1
    ):

        print(f"\nTrade #{number}")

        print(
            f"BUY:   "
            f"{trade['entry_date']}"
        )

        print(
            f"SELL:  "
            f"{trade['exit_date']}"
        )

        print(
            f"Entry: "
            f"${trade['entry_price']:.2f}"
        )

        print(
            f"Exit:  "
            f"${trade['exit_price']:.2f}"
        )

        print(
            f"Return: "
            f"{trade['return_pct']:.2f}%"
        )

else:

    print("\nNo completed trades.")