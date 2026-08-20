from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier

from TradingBot.archive.data import get_daily_data
from ml.features import build_features
from ml.market_features import build_market_features


# ============================================================
# SETTINGS
# ============================================================

SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
]

STARTING_CASH = 10_000

BUY_THRESHOLD = 0.65

HOLD_DAYS = 5
PURGE_DAYS = 5

SLIPPAGE = 0.0005

DATA_START = datetime(
    2010, 1, 1,
    tzinfo=timezone.utc
)

DATA_END = datetime.now(
    timezone.utc
)

TEST_YEARS = [
    2022,
    2023,
    2024,
    2025,
    2026,
]


# ============================================================
# FEATURES
# ============================================================

STOCK_FEATURES = [
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


MARKET_FEATURES = [
    "spy_return_5d",
    "spy_return_20d",

    "spy_vs_sma50",
    "spy_vs_sma200",

    "qqq_return_5d",
    "qqq_return_20d",

    "qqq_vs_sma50",
    "qqq_vs_sma200",

    "relative_spy_5d",
    "relative_spy_20d",

    "relative_qqq_5d",
    "relative_qqq_20d",
]


FEATURES = (
    STOCK_FEATURES
    + MARKET_FEATURES
)


# ============================================================
# MODEL
# ============================================================

def create_model():

    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42
    )


# ============================================================
# DOWNLOAD MARKET DATA
# ============================================================

print("\n============================================")
print("          DOWNLOADING MARKET DATA")
print("============================================\n")


print("Downloading SPY...")

spy_raw = get_daily_data(
    "SPY",
    DATA_START,
    DATA_END
)


print("Downloading QQQ...")

qqq_raw = get_daily_data(
    "QQQ",
    DATA_START,
    DATA_END
)


benchmark_data = {
    "SPY": spy_raw.copy(),
    "QQQ": qqq_raw.copy()
}


# ============================================================
# DOWNLOAD STOCKS + BUILD FEATURES
# ============================================================

all_symbol_data = {}

training_frames = []


for symbol in SYMBOLS:

    print(
        f"Downloading {symbol}..."
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


    combined["symbol_name"] = symbol


    all_symbol_data[
        symbol
    ] = combined.copy()


    training_frames.append(
        combined
    )


# ============================================================
# COMBINE DATASET
# ============================================================

dataset = pd.concat(
    training_frames,
    ignore_index=True
)


dataset = dataset.sort_values(
    "timestamp"
).reset_index(drop=True)


print(
    f"\nTotal ML samples: "
    f"{len(dataset)}"
)


print(
    f"Features per sample: "
    f"{len(FEATURES)}"
)


# ============================================================
# BENCHMARK RETURN
# ============================================================

def benchmark_return(
    df,
    start_time,
    end_time,
    starting_cash
):

    period = df[
        (
            df["timestamp"]
            >= start_time
        )
        &
        (
            df["timestamp"]
            <= end_time
        )
    ].copy()


    if len(period) < 2:
        return None


    first_price = (
        period.iloc[0]["open"]
    )

    last_price = (
        period.iloc[-1]["close"]
    )


    shares = int(
        starting_cash
        // first_price
    )


    leftover = (
        starting_cash
        - shares * first_price
    )


    ending_value = (
        leftover
        + shares * last_price
    )


    return (
        ending_value
        / starting_cash
        - 1
    ) * 100


# ============================================================
# DRAWDOWN
# ============================================================

def calculate_drawdown(
    equity_values
):

    if len(equity_values) == 0:
        return 0.0


    equity = pd.Series(
        equity_values,
        dtype=float
    )


    running_max = (
        equity.cummax()
    )


    drawdown = (
        equity
        / running_max
        - 1
    )


    return (
        drawdown.min()
        * 100
    )


# ============================================================
# TEST ONE YEAR
# ============================================================

def test_year(year):

    print("\n")
    print("=" * 65)
    print(f"TESTING YEAR {year}")
    print("=" * 65)


    test_start = pd.Timestamp(
        f"{year}-01-01",
        tz="UTC"
    )


    if (
        year
        == datetime.now().year
    ):

        test_end = pd.Timestamp.now(
            tz="UTC"
        )

    else:

        test_end = pd.Timestamp(
            f"{year}-12-31 23:59:59",
            tz="UTC"
        )


    # ========================================================
    # PURGE GAP
    # ========================================================

    prior_dates = (
        dataset[
            dataset["timestamp"]
            < test_start
        ]["timestamp"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )


    if len(prior_dates) <= PURGE_DAYS:

        print(
            "Not enough prior data."
        )

        return None


    purge_cutoff = (
        prior_dates.iloc[
            -(PURGE_DAYS + 1)
        ]
    )


    # ========================================================
    # TRAIN / TEST SETS
    # ========================================================

    train_df = dataset[
        dataset["timestamp"]
        <= purge_cutoff
    ].copy()


    test_df = dataset[
        (
            dataset["timestamp"]
            >= test_start
        )
        &
        (
            dataset["timestamp"]
            <= test_end
        )
    ].copy()


    if (
        len(train_df) == 0
        or
        len(test_df) == 0
    ):

        print(
            "Missing training or testing data."
        )

        return None


    print(
        f"Training samples: "
        f"{len(train_df)}"
    )

    print(
        f"Testing samples:  "
        f"{len(test_df)}"
    )

    print(
        f"Training cutoff:  "
        f"{purge_cutoff.date()}"
    )


    X_train = (
        train_df[FEATURES]
    )

    y_train = (
        train_df["target"]
    )


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    print(
        "Training model..."
    )


    model = create_model()


    model.fit(
        X_train,
        y_train
    )


    # ========================================================
    # GENERATE SIGNALS
    # ========================================================

    signals = []

    yearly_frames = {}


    for symbol in SYMBOLS:

        symbol_df = (
            all_symbol_data[
                symbol
            ]
        )


        year_df = symbol_df[
            (
                symbol_df["timestamp"]
                >= test_start
            )
            &
            (
                symbol_df["timestamp"]
                <= test_end
            )
        ].copy()


        year_df = (
            year_df
            .reset_index(drop=True)
        )


        if len(year_df) < 10:

            continue


        probabilities = (
            model.predict_proba(
                year_df[FEATURES]
            )[:, 1]
        )


        year_df[
            "prob_up"
        ] = probabilities


        yearly_frames[
            symbol
        ] = year_df


        for i in range(
            len(year_df)
        ):

            row = year_df.iloc[i]

            probability = (
                row["prob_up"]
            )


            # =================================================
            # NEW: MARKET REGIME RISK FILTER
            # =================================================
            #
            # Only allow LONG trades when both broad-market
            # ETFs are above their 200-day moving averages.
            #

            bull_market = (
                row["spy_vs_sma200"] > 0
                and
                row["qqq_vs_sma200"] > 0
            )


            if (
                probability >= BUY_THRESHOLD
                and
                bull_market
            ):

                signals.append({

                    "timestamp":
                        row["timestamp"],

                    "symbol":
                        symbol,

                    "prob_up":
                        probability,

                    "index":
                        i
                })


    # ========================================================
    # BENCHMARKS
    # ========================================================

    spy_return = benchmark_return(
        benchmark_data["SPY"],
        test_start,
        test_end,
        STARTING_CASH
    )


    qqq_return = benchmark_return(
        benchmark_data["QQQ"],
        test_start,
        test_end,
        STARTING_CASH
    )


    # ========================================================
    # NO SIGNALS
    # ========================================================

    if len(signals) == 0:

        print(
            "No qualifying signals."
        )

        return {

            "year":
                year,

            "return":
                0,

            "drawdown":
                0,

            "trades":
                0,

            "win_rate":
                0,

            "avg_trade":
                0,

            "spy":
                spy_return,

            "qqq":
                qqq_return
        }


    # ========================================================
    # SIGNAL TABLE
    # ========================================================

    signals = pd.DataFrame(
        signals
    )


    signals = signals.sort_values(

        [
            "timestamp",
            "prob_up"
        ],

        ascending=[
            True,
            False
        ]

    ).reset_index(
        drop=True
    )


    print(
        f"Signals >= "
        f"{BUY_THRESHOLD:.0%} "
        f"after regime filter: "
        f"{len(signals)}"
    )


    # ========================================================
    # PORTFOLIO
    # ========================================================

    cash = STARTING_CASH

    trades = []

    equity_history = [
        STARTING_CASH
    ]

    busy_until = None


    # ========================================================
    # EXECUTE TRADES
    # ========================================================

    for signal_time in sorted(
        signals[
            "timestamp"
        ].unique()
    ):

        signal_time = pd.Timestamp(
            signal_time
        )


        if (
            busy_until is not None
            and
            signal_time <= busy_until
        ):

            continue


        daily_signals = signals[
            signals["timestamp"]
            == signal_time
        ]


        best = (
            daily_signals
            .iloc[0]
        )


        symbol = (
            best["symbol"]
        )


        probability = (
            best["prob_up"]
        )


        signal_index = int(
            best["index"]
        )


        symbol_df = (
            yearly_frames[
                symbol
            ]
        )


        # ====================================================
        # ENTRY / EXIT
        # ====================================================

        entry_index = (
            signal_index + 1
        )


        exit_index = (
            entry_index
            + HOLD_DAYS
        )


        if (
            exit_index
            >= len(symbol_df)
        ):

            continue


        entry_row = (
            symbol_df.iloc[
                entry_index
            ]
        )


        exit_row = (
            symbol_df.iloc[
                exit_index
            ]
        )


        # ====================================================
        # PRICES + SLIPPAGE
        # ====================================================

        entry_price = (
            entry_row["open"]
            * (
                1
                + SLIPPAGE
            )
        )


        exit_price = (
            exit_row["open"]
            * (
                1
                - SLIPPAGE
            )
        )


        # ====================================================
        # POSITION SIZE
        # ====================================================

        shares = int(
            cash
            // entry_price
        )


        if shares <= 0:

            continue


        leftover = (
            cash
            - shares
            * entry_price
        )


        # ====================================================
        # DAILY MARK-TO-MARKET
        # ====================================================

        for day_index in range(
            entry_index,
            exit_index + 1
        ):

            row = (
                symbol_df.iloc[
                    day_index
                ]
            )


            equity_value = (
                leftover
                + shares
                * row["close"]
            )


            equity_history.append(
                equity_value
            )


        # ====================================================
        # EXIT
        # ====================================================

        sale_value = (
            shares
            * exit_price
        )


        new_cash = (
            leftover
            + sale_value
        )


        profit = (
            new_cash
            - cash
        )


        return_pct = (
            new_cash
            / cash
            - 1
        ) * 100


        trades.append({

            "symbol":
                symbol,

            "probability":
                probability,

            "return_pct":
                return_pct,

            "profit":
                profit,

            "entry_date":
                entry_row["timestamp"],

            "exit_date":
                exit_row["timestamp"]
        })


        cash = new_cash


        equity_history.append(
            cash
        )


        busy_until = (
            exit_row[
                "timestamp"
            ]
        )


    # ========================================================
    # METRICS
    # ========================================================

    total_return = (
        cash
        / STARTING_CASH
        - 1
    ) * 100


    max_drawdown = (
        calculate_drawdown(
            equity_history
        )
    )


    if len(trades) > 0:

        winners = [
            trade
            for trade in trades
            if trade["profit"] > 0
        ]


        win_rate = (
            len(winners)
            / len(trades)
        ) * 100


        avg_trade = np.mean(
            [
                trade[
                    "return_pct"
                ]
                for trade
                in trades
            ]
        )

    else:

        win_rate = 0
        avg_trade = 0


    # ========================================================
    # YEAR REPORT
    # ========================================================

    print("\n-----------------------------")
    print("YEAR RESULTS")
    print("-----------------------------")


    print(
        f"AI Return:      "
        f"{total_return:.2f}%"
    )

    print(
        f"Max Drawdown:   "
        f"{max_drawdown:.2f}%"
    )

    print(
        f"Trades:         "
        f"{len(trades)}"
    )

    print(
        f"Win Rate:       "
        f"{win_rate:.2f}%"
    )

    print(
        f"Average Trade:  "
        f"{avg_trade:.2f}%"
    )

    print(
        f"SPY Return:     "
        f"{spy_return:.2f}%"
    )

    print(
        f"QQQ Return:     "
        f"{qqq_return:.2f}%"
    )


    return {

        "year":
            year,

        "return":
            total_return,

        "drawdown":
            max_drawdown,

        "trades":
            len(trades),

        "win_rate":
            win_rate,

        "avg_trade":
            avg_trade,

        "spy":
            spy_return,

        "qqq":
            qqq_return
    }


# ============================================================
# RUN WALK-FORWARD
# ============================================================

year_results = []


for year in TEST_YEARS:

    result = test_year(
        year
    )

    if result is not None:

        year_results.append(
            result
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n\n")

print("=" * 85)

print(
    "        V3 WALK-FORWARD + RISK FILTER"
)

print("=" * 85)


print(

    f"{'YEAR':<8}"

    f"{'AI':>11}"

    f"{'SPY':>11}"

    f"{'QQQ':>11}"

    f"{'DD':>11}"

    f"{'TRADES':>10}"

    f"{'WIN%':>10}"
)


print("-" * 85)


for result in year_results:

    print(

        f"{result['year']:<8}"

        f"{result['return']:>10.2f}%"

        f"{result['spy']:>10.2f}%"

        f"{result['qqq']:>10.2f}%"

        f"{result['drawdown']:>10.2f}%"

        f"{result['trades']:>10}"

        f"{result['win_rate']:>9.2f}%"
    )


print("=" * 85)


# ============================================================
# COMPOUND RESULT
# ============================================================

portfolio = STARTING_CASH


for result in year_results:

    portfolio *= (
        1
        + result[
            "return"
        ] / 100
    )


combined_return = (
    portfolio
    / STARTING_CASH
    - 1
) * 100


print(
    f"\nStarting Capital: "
    f"${STARTING_CASH:,.2f}"
)

print(
    f"V3 Ending Value:  "
    f"${portfolio:,.2f}"
)

print(
    f"V3 Combined Return: "
    f"{combined_return:.2f}%"
)