from datetime import datetime, timezone

import joblib
import pandas as pd

from TradingBot.archive.data import get_daily_data
from ml.features import build_features


# ==========================================
# SETTINGS
# ==========================================

SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
]

FEATURES = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",

    "price_vs_sma10",
    "price_vs_sma20",
    "price_vs_sma50",
    "price_vs_sma100",

    "volatility_10",
    "volatility_20",

    "rsi_14",
    "volume_ratio",
    "daily_range",
]

STARTING_CASH = 10_000

# Only trade when model confidence exceeds this
BUY_THRESHOLD = 0.65

# Hold each position for 5 trading days
HOLD_DAYS = 5

# 0.05% simulated slippage
SLIPPAGE = 0.0005

# We only allow ONE position at a time for now.
TEST_START = pd.Timestamp(
    "2025-01-01",
    tz="UTC"
)

TEST_END = datetime.now(
    timezone.utc
)

# Extra history is needed to calculate indicators.
DATA_START = datetime(
    2024, 6, 1,
    tzinfo=timezone.utc
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

print("\nLoading trained AI model...")

model = joblib.load(
    "ml/trading_model.joblib"
)

print("Model loaded.")


# ==========================================
# DOWNLOAD DATA
# ==========================================

all_data = {}

print("\nDownloading market data...\n")

for symbol in SYMBOLS:

    print(f"Processing {symbol}...")

    raw = get_daily_data(
        symbol,
        DATA_START,
        TEST_END
    )

    df = build_features(raw)

    df = df[
        df["timestamp"] >= TEST_START
    ].copy()

    df = df.reset_index(drop=True)

    # Generate AI probabilities
    df["prob_up"] = model.predict_proba(
        df[FEATURES]
    )[:, 1]

    all_data[symbol] = df


# ==========================================
# CREATE SIGNAL TABLE
# ==========================================

signals = []

for symbol, df in all_data.items():

    for i in range(len(df) - HOLD_DAYS - 1):

        row = df.iloc[i]

        if row["prob_up"] >= BUY_THRESHOLD:

            signals.append({
                "timestamp": row["timestamp"],
                "symbol": symbol,
                "prob_up": row["prob_up"],
                "index": i,
            })


signals = pd.DataFrame(signals)

signals = signals.sort_values(
    ["timestamp", "prob_up"],
    ascending=[True, False]
).reset_index(drop=True)


print(
    f"\nFound {len(signals)} candidate "
    f"signals >= {BUY_THRESHOLD:.0%}."
)


# ==========================================
# PORTFOLIO
# ==========================================

cash = STARTING_CASH

trades = []

equity_history = [
    STARTING_CASH
]

last_exit_time = None


# ==========================================
# EXECUTE TRADES
# ==========================================

for timestamp in sorted(
    signals["timestamp"].unique()
):

    timestamp = pd.Timestamp(timestamp)

    # We only allow one position at a time.
    if (
        last_exit_time is not None
        and timestamp <= last_exit_time
    ):
        continue

    daily_signals = signals[
        signals["timestamp"] == timestamp
    ]

    # Pick the stock with the highest probability.
    best_signal = daily_signals.iloc[0]

    symbol = best_signal["symbol"]
    probability = best_signal["prob_up"]
    signal_index = int(best_signal["index"])

    df = all_data[symbol]

    # Signal generated using today's close.
    #
    # Therefore we cannot buy until tomorrow.
    entry_index = signal_index + 1

    # Hold for HOLD_DAYS trading sessions.
    exit_index = entry_index + HOLD_DAYS

    if exit_index >= len(df):
        continue

    entry_row = df.iloc[entry_index]
    exit_row = df.iloc[exit_index]

    # Buy next day's OPEN.
    entry_price = (
        entry_row["open"]
        * (1 + SLIPPAGE)
    )

    # Sell at exit day's OPEN.
    exit_price = (
        exit_row["open"]
        * (1 - SLIPPAGE)
    )

    shares = int(
        cash // entry_price
    )

    if shares <= 0:
        continue

    leftover_cash = (
        cash
        - shares * entry_price
    )

    sale_value = (
        shares * exit_price
    )

    new_cash = (
        leftover_cash
        + sale_value
    )

    profit = (
        new_cash - cash
    )

    return_pct = (
        new_cash / cash - 1
    ) * 100

    trades.append({
        "symbol": symbol,

        "signal_date":
            timestamp,

        "entry_date":
            entry_row["timestamp"],

        "exit_date":
            exit_row["timestamp"],

        "probability":
            probability,

        "entry_price":
            entry_price,

        "exit_price":
            exit_price,

        "shares":
            shares,

        "profit":
            profit,

        "return_pct":
            return_pct,

        "portfolio_before":
            cash,

        "portfolio_after":
            new_cash,
    })

    cash = new_cash

    equity_history.append(
        cash
    )

    last_exit_time = (
        exit_row["timestamp"]
    )


# ==========================================
# RESULTS
# ==========================================

if len(trades) > 0:

    trade_df = pd.DataFrame(trades)

    winners = trade_df[
        trade_df["profit"] > 0
    ]

    win_rate = (
        len(winners)
        / len(trade_df)
        * 100
    )

    avg_trade = (
        trade_df["return_pct"]
        .mean()
    )

    best_trade = (
        trade_df["return_pct"]
        .max()
    )

    worst_trade = (
        trade_df["return_pct"]
        .min()
    )

else:

    trade_df = pd.DataFrame()

    win_rate = 0
    avg_trade = 0
    best_trade = 0
    worst_trade = 0


bot_return = (
    cash / STARTING_CASH - 1
) * 100


# ==========================================
# TRADE-LEVEL DRAWDOWN
# ==========================================

equity = pd.Series(
    equity_history
)

running_max = (
    equity.cummax()
)

drawdown = (
    equity / running_max - 1
)

max_drawdown = (
    drawdown.min() * 100
)


# ==========================================
# PRINT RESULTS
# ==========================================

print("\n==========================================")
print("            AI BACKTEST")
print("==========================================")

print(
    f"\nStarting Capital: "
    f"${STARTING_CASH:,.2f}"
)

print(
    f"Ending Capital:   "
    f"${cash:,.2f}"
)

print(
    f"\nAI Return:        "
    f"{bot_return:.2f}%"
)

print(
    f"Max Drawdown*:    "
    f"{max_drawdown:.2f}%"
)

print(
    f"\nCompleted Trades: "
    f"{len(trades)}"
)

print(
    f"Win Rate:         "
    f"{win_rate:.2f}%"
)

print(
    f"Average Trade:    "
    f"{avg_trade:.2f}%"
)

print(
    f"Best Trade:       "
    f"{best_trade:.2f}%"
)

print(
    f"Worst Trade:      "
    f"{worst_trade:.2f}%"
)

print(
    "\n*Drawdown currently measured "
    "between completed trades."
)


# ==========================================
# PRINT TRADES
# ==========================================

if len(trade_df) > 0:

    print("\n==========================================")
    print("                TRADES")
    print("==========================================")

    for _, trade in trade_df.iterrows():

        print(
            f"\n{trade['symbol']} | "
            f"AI confidence: "
            f"{trade['probability']:.1%}"
        )

        print(
            f"Signal: "
            f"{trade['signal_date']}"
        )

        print(
            f"BUY:    "
            f"{trade['entry_date']} "
            f"@ ${trade['entry_price']:.2f}"
        )

        print(
            f"SELL:   "
            f"{trade['exit_date']} "
            f"@ ${trade['exit_price']:.2f}"
        )

        print(
            f"Return: "
            f"{trade['return_pct']:.2f}%"
        )

        print(
            f"Profit: "
            f"${trade['profit']:.2f}"
        )

        print(
            f"Account: "
            f"${trade['portfolio_after']:,.2f}"
        )