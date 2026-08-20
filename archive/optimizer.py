from datetime import datetime, timezone

from TradingBot.archive.data import get_daily_data
from strategies.sma_cross import add_signals
from TradingBot.archive.backtester import run_backtest
from TradingBot.archive.metrics import calculate_metrics


SYMBOL = "AAPL"
STARTING_CASH = 10_000

# IMPORTANT:
# We're only optimizing on 2021-2024.
# 2025+ stays hidden for later testing.
TRAIN_START = datetime(2021, 1, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2025, 1, 1, tzinfo=timezone.utc)


print("Downloading training data...")

df_original = get_daily_data(
    SYMBOL,
    TRAIN_START,
    TRAIN_END
)

print(f"Downloaded {len(df_original)} trading days.")
print("\nTesting strategies...\n")


# Parameters we'll test
FAST_VALUES = [
    5,
    10,
    15,
    20,
    25,
    30
]

SLOW_VALUES = [
    30,
    40,
    50,
    75,
    100,
    150,
    200
]

TREND_VALUES = [
    100,
    150,
    200
]


results = []


for fast in FAST_VALUES:

    for slow in SLOW_VALUES:

        # Fast average must actually be faster
        if fast >= slow:
            continue

        for trend in TREND_VALUES:

            df = add_signals(
                df_original,
                fast=fast,
                slow=slow,
                trend_filter=trend
            )

            equity, trades = run_backtest(
                df,
                starting_cash=STARTING_CASH
            )

            metrics = calculate_metrics(
                equity,
                trades,
                STARTING_CASH
            )

            results.append({
                "fast": fast,
                "slow": slow,
                "trend": trend,

                "return":
                    metrics["total_return"],

                "drawdown":
                    metrics["max_drawdown"],

                "win_rate":
                    metrics["win_rate"],

                "trades":
                    metrics["trades"]
            })


# Sort by highest return
results.sort(
    key=lambda x: x["return"],
    reverse=True
)


print("=" * 70)
print("TOP 10 STRATEGIES")
print("=" * 70)


for i, result in enumerate(results[:10], start=1):

    print(
        f"{i:2}. "
        f"SMA {result['fast']:3}/{result['slow']:3} "
        f"Trend {result['trend']:3} | "
        f"Return: {result['return']:7.2f}% | "
        f"DD: {result['drawdown']:7.2f}% | "
        f"Win: {result['win_rate']:6.2f}% | "
        f"Trades: {result['trades']:2}"
    )