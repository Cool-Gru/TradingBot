import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed


# ==========================================
# SETTINGS
# ==========================================

SYMBOL = "AAPL"
STARTING_CASH = 10_000

FAST_SMA = 20
SLOW_SMA = 50

# Simulated slippage per transaction
SLIPPAGE = 0.0005  # 0.05%


# ==========================================
# CONNECT TO ALPACA
# ==========================================

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

client = StockHistoricalDataClient(
    API_KEY,
    SECRET_KEY
)


# ==========================================
# DOWNLOAD DATA
# ==========================================

end = datetime.now(timezone.utc)
start = end - timedelta(days=365 * 5)

request = StockBarsRequest(
    symbol_or_symbols=SYMBOL,
    timeframe=TimeFrame.Day,
    start=start,
    end=end,
    feed=DataFeed.IEX
)

bars = client.get_stock_bars(request)

df = bars.df.reset_index()

# Just in case multiple symbols are added later
if "symbol" in df.columns:
    df = df[df["symbol"] == SYMBOL].copy()

df = df.sort_values("timestamp").reset_index(drop=True)


# ==========================================
# INDICATORS
# ==========================================

df["SMA_FAST"] = df["close"].rolling(FAST_SMA).mean()
df["SMA_SLOW"] = df["close"].rolling(SLOW_SMA).mean()

df = df.dropna().reset_index(drop=True)


# ==========================================
# BACKTEST
# ==========================================

cash = STARTING_CASH
shares = 0

entry_price = None
entry_date = None

trades = []
equity_history = []


for i in range(1, len(df) - 1):

    yesterday = df.iloc[i - 1]
    today = df.iloc[i]
    tomorrow = df.iloc[i + 1]

    # --------------------------------------
    # Detect crossover using TODAY'S close
    # --------------------------------------

    crossed_up = (
        yesterday["SMA_FAST"] <= yesterday["SMA_SLOW"]
        and today["SMA_FAST"] > today["SMA_SLOW"]
    )

    crossed_down = (
        yesterday["SMA_FAST"] >= yesterday["SMA_SLOW"]
        and today["SMA_FAST"] < today["SMA_SLOW"]
    )

    # --------------------------------------
    # BUY tomorrow at open
    # --------------------------------------

    if crossed_up and shares == 0:

        execution_price = tomorrow["open"] * (1 + SLIPPAGE)

        shares_to_buy = int(cash // execution_price)

        if shares_to_buy > 0:

            cost = shares_to_buy * execution_price

            cash -= cost
            shares = shares_to_buy

            entry_price = execution_price
            entry_date = tomorrow["timestamp"]

    # --------------------------------------
    # SELL tomorrow at open
    # --------------------------------------

    elif crossed_down and shares > 0:

        execution_price = tomorrow["open"] * (1 - SLIPPAGE)

        cash += shares * execution_price

        profit = (
            execution_price - entry_price
        ) * shares

        return_pct = (
            execution_price / entry_price - 1
        ) * 100

        trades.append({
            "entry_date": entry_date,
            "exit_date": tomorrow["timestamp"],
            "entry_price": entry_price,
            "exit_price": execution_price,
            "shares": shares,
            "profit": profit,
            "return_pct": return_pct
        })

        shares = 0
        entry_price = None
        entry_date = None

    # --------------------------------------
    # Record portfolio value
    # --------------------------------------

    portfolio_value = (
        cash + shares * today["close"]
    )

    equity_history.append(portfolio_value)


# ==========================================
# FINAL PORTFOLIO VALUE
# ==========================================

final_price = df.iloc[-1]["close"]

final_value = cash + shares * final_price

bot_return = (
    (final_value / STARTING_CASH) - 1
) * 100


# ==========================================
# BUY & HOLD
# ==========================================

first_price = df.iloc[0]["open"]

buy_hold_shares = int(
    STARTING_CASH // first_price
)

buy_hold_cash = (
    STARTING_CASH -
    buy_hold_shares * first_price
)

buy_hold_value = (
    buy_hold_cash +
    buy_hold_shares * final_price
)

buy_hold_return = (
    (buy_hold_value / STARTING_CASH) - 1
) * 100


# ==========================================
# TRADE STATISTICS
# ==========================================

trade_df = pd.DataFrame(trades)

if len(trade_df) > 0:

    winners = trade_df[
        trade_df["profit"] > 0
    ]

    win_rate = (
        len(winners) /
        len(trade_df)
    ) * 100

    average_trade = (
        trade_df["return_pct"].mean()
    )

    best_trade = (
        trade_df["return_pct"].max()
    )

    worst_trade = (
        trade_df["return_pct"].min()
    )

else:

    win_rate = 0
    average_trade = 0
    best_trade = 0
    worst_trade = 0


# ==========================================
# MAXIMUM DRAWDOWN
# ==========================================

equity = pd.Series(equity_history)

running_max = equity.cummax()

drawdown = (
    equity - running_max
) / running_max

max_drawdown = (
    drawdown.min() * 100
)


# ==========================================
# RESULTS
# ==========================================

print("\n===================================")
print("        BACKTESTER V2")
print("===================================")

print(f"\nSymbol:             {SYMBOL}")

print(
    f"Strategy:           "
    f"SMA {FAST_SMA}/{SLOW_SMA}"
)

print(
    f"Starting Capital:   "
    f"${STARTING_CASH:,.2f}"
)

print(
    f"Ending Capital:     "
    f"${final_value:,.2f}"
)

print("\n---------------")

print(
    f"Bot Return:         "
    f"{bot_return:.2f}%"
)

print(
    f"Buy & Hold:         "
    f"{buy_hold_return:.2f}%"
)

print(
    f"Max Drawdown:       "
    f"{max_drawdown:.2f}%"
)

print("\n---------------")

print(
    f"Completed Trades:   "
    f"{len(trades)}"
)

print(
    f"Win Rate:           "
    f"{win_rate:.2f}%"
)

print(
    f"Average Trade:      "
    f"{average_trade:.2f}%"
)

print(
    f"Best Trade:         "
    f"{best_trade:.2f}%"
)

print(
    f"Worst Trade:        "
    f"{worst_trade:.2f}%"
)


# ==========================================
# PRINT INDIVIDUAL TRADES
# ==========================================

if len(trade_df) > 0:

    print("\n===================================")
    print("              TRADES")
    print("===================================")

    for _, trade in trade_df.iterrows():

        print(
            f"\nBUY:  {trade['entry_date']}"
        )

        print(
            f"SELL: {trade['exit_date']}"
        )

        print(
            f"Entry: ${trade['entry_price']:.2f}"
        )

        print(
            f"Exit:  ${trade['exit_price']:.2f}"
        )

        print(
            f"Return: {trade['return_pct']:.2f}%"
        )

        print(
            f"Profit: ${trade['profit']:.2f}"
        )