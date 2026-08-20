from datetime import datetime, timezone

from TradingBot.archive.data import get_daily_data
from strategies.sma_cross import add_signals
from TradingBot.archive.backtester import run_backtest
from TradingBot.archive.metrics import calculate_metrics


SYMBOL = "AAPL"

STARTING_CASH = 10000


start = datetime(
    2021,
    1,
    1,
    tzinfo=timezone.utc
)

end = datetime.now(
    timezone.utc
)


print("Downloading data...")

df = get_daily_data(
    SYMBOL,
    start,
    end
)

print(
    f"Downloaded {len(df)} days."
)


print("Calculating signals...")

df = add_signals(
    df,
    fast=20,
    slow=50,
    trend_filter=200
)


print("Running backtest...")

equity_history, trades = (
    run_backtest(
        df,
        starting_cash=STARTING_CASH
    )
)


results = calculate_metrics(
    equity_history,
    trades,
    STARTING_CASH
)


print("\n==============================")
print("       STRATEGY REPORT")
print("==============================")

print(f"\nSymbol: {SYMBOL}")

print(
    f"Ending Value: "
    f"${results['final_value']:,.2f}"
)

print(
    f"Return: "
    f"{results['total_return']:.2f}%"
)

print(
    f"Max Drawdown: "
    f"{results['max_drawdown']:.2f}%"
)

print(
    f"Win Rate: "
    f"{results['win_rate']:.2f}%"
)

print(
    f"Average Trade: "
    f"{results['avg_trade']:.2f}%"
)

print(
    f"Best Trade: "
    f"{results['best_trade']:.2f}%"
)

print(
    f"Worst Trade: "
    f"{results['worst_trade']:.2f}%"
)

print(
    f"Completed Trades: "
    f"{results['trades']}"
)