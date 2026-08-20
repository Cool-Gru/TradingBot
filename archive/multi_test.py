from datetime import datetime, timezone

from TradingBot.archive.data import get_daily_data
from strategies.sma_cross import add_signals
from TradingBot.archive.backtester import run_backtest
from TradingBot.archive.metrics import calculate_metrics


STARTING_CASH = 10_000

FAST = 30
SLOW = 50
TREND = 100

SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "SPY",
    "QQQ"
]


# Extra data for indicator warm-up
DATA_START = datetime(
    2024, 6, 1,
    tzinfo=timezone.utc
)

# Actual test period
TEST_START = datetime(
    2025, 1, 1,
    tzinfo=timezone.utc
)

TEST_END = datetime.now(
    timezone.utc
)


results_table = []


print("\n==============================================")
print("           MULTI-STOCK VALIDATION")
print("==============================================")

print(
    f"\nStrategy: SMA {FAST}/{SLOW} "
    f"Trend {TREND}"
)

print(
    f"Test period: {TEST_START.date()} "
    f"to {TEST_END.date()}"
)


for symbol in SYMBOLS:

    print(f"\nTesting {symbol}...")

    try:

        # ======================================
        # DOWNLOAD DATA
        # ======================================

        df = get_daily_data(
            symbol,
            DATA_START,
            TEST_END
        )


        # ======================================
        # CALCULATE SIGNALS
        # ======================================

        df = add_signals(
            df,
            fast=FAST,
            slow=SLOW,
            trend_filter=TREND
        )


        # ======================================
        # REMOVE WARM-UP PERIOD
        # ======================================

        df = df[
            df["timestamp"] >= TEST_START
        ].copy()

        df = df.reset_index(drop=True)


        # Safety check
        if len(df) < 2:

            print(
                f"Not enough data for {symbol}"
            )

            continue


        # ======================================
        # RUN BOT
        # ======================================

        equity, trades = run_backtest(
            df,
            starting_cash=STARTING_CASH
        )

        metrics = calculate_metrics(
            equity,
            trades,
            STARTING_CASH
        )


        # ======================================
        # BUY & HOLD
        # ======================================

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
            buy_hold_value
            / STARTING_CASH
            - 1
        ) * 100


        # ======================================
        # DIFFERENCE VS BUY & HOLD
        # ======================================

        excess_return = (
            metrics["total_return"]
            - buy_hold_return
        )


        results_table.append({

            "symbol":
                symbol,

            "bot_return":
                metrics["total_return"],

            "buy_hold":
                buy_hold_return,

            "excess":
                excess_return,

            "drawdown":
                metrics["max_drawdown"],

            "win_rate":
                metrics["win_rate"],

            "trades":
                metrics["trades"]
        })


    except Exception as e:

        print(
            f"Error testing {symbol}: {e}"
        )


# ==========================================
# SORT RESULTS
# ==========================================

results_table.sort(
    key=lambda x: x["excess"],
    reverse=True
)


# ==========================================
# PRINT TABLE
# ==========================================

print("\n")
print("=" * 95)

print(
    f"{'SYMBOL':<8}"
    f"{'BOT':>12}"
    f"{'BUY&HOLD':>14}"
    f"{'EXCESS':>12}"
    f"{'MAX DD':>12}"
    f"{'WIN RATE':>12}"
    f"{'TRADES':>10}"
)

print("=" * 95)


for result in results_table:

    print(
        f"{result['symbol']:<8}"

        f"{result['bot_return']:>11.2f}%"

        f"{result['buy_hold']:>13.2f}%"

        f"{result['excess']:>11.2f}%"

        f"{result['drawdown']:>11.2f}%"

        f"{result['win_rate']:>11.2f}%"

        f"{result['trades']:>10}"
    )


print("=" * 95)


# ==========================================
# SUMMARY
# ==========================================

if len(results_table) > 0:

    beat_market = sum(
        1
        for result in results_table
        if result["excess"] > 0
    )

    profitable = sum(
        1
        for result in results_table
        if result["bot_return"] > 0
    )

    average_bot_return = sum(
        result["bot_return"]
        for result in results_table
    ) / len(results_table)

    average_buy_hold = sum(
        result["buy_hold"]
        for result in results_table
    ) / len(results_table)


    print("\n============== SUMMARY ==============")

    print(
        f"Profitable on: "
        f"{profitable}/{len(results_table)} symbols"
    )

    print(
        f"Beat Buy & Hold on: "
        f"{beat_market}/{len(results_table)} symbols"
    )

    print(
        f"Average Bot Return: "
        f"{average_bot_return:.2f}%"
    )

    print(
        f"Average Buy & Hold: "
        f"{average_buy_hold:.2f}%"
    )