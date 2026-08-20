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


STARTING_CASH = 10_000

BUY_THRESHOLD = 0.65

HOLD_DAYS = 5
PURGE_DAYS = 5

SLIPPAGE = 0.0005


DATA_START = datetime(
    2010,
    1,
    1,
    tzinfo=timezone.utc
)

DATA_END = datetime.now(
    timezone.utc
)


# Half-year test windows
WINDOWS = [

    (
        "2022 H1",
        "2022-01-01",
        "2022-06-30"
    ),

    (
        "2022 H2",
        "2022-07-01",
        "2022-12-31"
    ),

    (
        "2023 H1",
        "2023-01-01",
        "2023-06-30"
    ),

    (
        "2023 H2",
        "2023-07-01",
        "2023-12-31"
    ),

    (
        "2024 H1",
        "2024-01-01",
        "2024-06-30"
    ),

    (
        "2024 H2",
        "2024-07-01",
        "2024-12-31"
    ),

    (
        "2025 H1",
        "2025-01-01",
        "2025-06-30"
    ),

    (
        "2025 H2",
        "2025-07-01",
        "2025-12-31"
    ),

    (
        "2026 H1",
        "2026-01-01",
        "2026-06-30"
    ),

    (
        "2026 H2",
        "2026-07-01",
        "2026-12-31"
    ),
]


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
print("         ROLLING VALIDATION SETUP")
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
# DOWNLOAD STOCK DATA
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


    stock_features = (
        build_features(
            raw
        )
    )


    combined = (
        build_market_features(
            stock_features,
            spy_raw,
            qqq_raw
        )
    )


    combined["symbol_name"] = (
        symbol
    )


    all_symbol_data[
        symbol
    ] = combined.copy()


    training_frames.append(
        combined
    )


dataset = pd.concat(
    training_frames,
    ignore_index=True
)


dataset = dataset.sort_values(
    "timestamp"
).reset_index(drop=True)


print(
    f"\nTotal samples: "
    f"{len(dataset)}"
)


# ============================================================
# BENCHMARK
# ============================================================

def benchmark_return(
    df,
    start_time,
    end_time
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


    return (
        last_price
        / first_price
        - 1
    ) * 100


# ============================================================
# DRAWDOWN
# ============================================================

def calculate_drawdown(
    equity_history
):

    if not equity_history:
        return 0.0


    equity = pd.Series(
        equity_history,
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
# TEST WINDOW
# ============================================================

def test_window(
    label,
    start_date,
    end_date
):

    print("\n")
    print("=" * 70)

    print(
        f"TEST WINDOW: {label}"
    )

    print("=" * 70)


    test_start = pd.Timestamp(
        start_date,
        tz="UTC"
    )


    requested_end = pd.Timestamp(
        end_date,
        tz="UTC"
    )


    now = pd.Timestamp.now(
        tz="UTC"
    )


    test_end = min(
        requested_end,
        now
    )


    if test_start > now:

        print(
            "Window is in the future."
        )

        return None


    # ========================================================
    # PURGE TRAINING DATA
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


    if (
        len(prior_dates)
        <= PURGE_DAYS
    ):

        print(
            "Not enough training history."
        )

        return None


    purge_cutoff = (
        prior_dates.iloc[
            -(PURGE_DAYS + 1)
        ]
    )


    train_df = dataset[
        dataset["timestamp"]
        <= purge_cutoff
    ].copy()


    print(
        f"Train samples: "
        f"{len(train_df)}"
    )

    print(
        f"Train cutoff:  "
        f"{purge_cutoff.date()}"
    )


    X_train = (
        train_df[
            FEATURES
        ]
    )

    y_train = (
        train_df[
            "target"
        ]
    )


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    model = create_model()


    model.fit(
        X_train,
        y_train
    )


    # ========================================================
    # BUILD SIGNALS
    # ========================================================

    signals = []

    window_frames = {}


    for symbol in SYMBOLS:

        full_df = (
            all_symbol_data[
                symbol
            ]
        )


        test_df = full_df[
            (
                full_df["timestamp"]
                >= test_start
            )
            &
            (
                full_df["timestamp"]
                <= test_end
            )
        ].copy()


        test_df = (
            test_df
            .reset_index(drop=True)
        )


        if len(test_df) < 10:
            continue


        test_df[
            "prob_up"
        ] = model.predict_proba(
            test_df[
                FEATURES
            ]
        )[:, 1]


        window_frames[
            symbol
        ] = test_df


        for i in range(
            len(test_df)
        ):

            row = (
                test_df.iloc[i]
            )


            bull_market = (
                row[
                    "spy_vs_sma200"
                ] > 0
                and
                row[
                    "qqq_vs_sma200"
                ] > 0
            )


            if (
                row["prob_up"]
                >= BUY_THRESHOLD
                and
                bull_market
            ):

                signals.append({
                    "timestamp":
                        row["timestamp"],

                    "symbol":
                        symbol,

                    "probability":
                        row["prob_up"],

                    "index":
                        i
                })


    if len(signals) == 0:

        print(
            "No valid signals."
        )

        return {
            "label": label,
            "return": 0.0,
            "drawdown": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "spy": benchmark_return(
                benchmark_data[
                    "SPY"
                ],
                test_start,
                test_end
            ),
            "qqq": benchmark_return(
                benchmark_data[
                    "QQQ"
                ],
                test_start,
                test_end
            )
        }


    signals = pd.DataFrame(
        signals
    )


    signals = signals.sort_values(
        [
            "timestamp",
            "probability"
        ],
        ascending=[
            True,
            False
        ]
    ).reset_index(drop=True)


    # ========================================================
    # PORTFOLIO
    # ========================================================

    cash = STARTING_CASH

    trades = []

    equity_history = [
        STARTING_CASH
    ]

    busy_until = None


    for signal_time in sorted(
        signals[
            "timestamp"
        ].unique()
    ):

        signal_time = (
            pd.Timestamp(
                signal_time
            )
        )


        if (
            busy_until is not None
            and
            signal_time <= busy_until
        ):

            continue


        daily = signals[
            signals["timestamp"]
            == signal_time
        ]


        best = (
            daily.iloc[0]
        )


        symbol = (
            best["symbol"]
        )


        i = int(
            best["index"]
        )


        df = (
            window_frames[
                symbol
            ]
        )


        entry_index = (
            i + 1
        )


        exit_index = (
            entry_index
            + HOLD_DAYS
        )


        if exit_index >= len(df):
            continue


        entry_row = (
            df.iloc[
                entry_index
            ]
        )


        exit_row = (
            df.iloc[
                exit_index
            ]
        )


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


        # Daily mark-to-market
        for day in range(
            entry_index,
            exit_index + 1
        ):

            row = df.iloc[
                day
            ]


            equity_history.append(
                leftover
                + shares
                * row["close"]
            )


        new_cash = (
            leftover
            + shares
            * exit_price
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

            "return_pct":
                return_pct,

            "profit":
                profit
        })


        cash = (
            new_cash
        )


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


    if trades:

        winners = [
            trade
            for trade in trades
            if trade[
                "profit"
            ] > 0
        ]


        win_rate = (
            len(winners)
            / len(trades)
            * 100
        )


        avg_trade = np.mean(
            [
                trade[
                    "return_pct"
                ]
                for trade in trades
            ]
        )


    else:

        win_rate = 0.0
        avg_trade = 0.0


    spy_return = (
        benchmark_return(
            benchmark_data[
                "SPY"
            ],
            test_start,
            test_end
        )
    )


    qqq_return = (
        benchmark_return(
            benchmark_data[
                "QQQ"
            ],
            test_start,
            test_end
        )
    )


    print("\n----------------------------")
    print("WINDOW RESULTS")
    print("----------------------------")


    print(
        f"AI Return:     "
        f"{total_return:.2f}%"
    )

    print(
        f"Max Drawdown:  "
        f"{max_drawdown:.2f}%"
    )

    print(
        f"Trades:        "
        f"{len(trades)}"
    )

    print(
        f"Win Rate:      "
        f"{win_rate:.2f}%"
    )

    print(
        f"Average Trade: "
        f"{avg_trade:.2f}%"
    )

    print(
        f"SPY:           "
        f"{spy_return:.2f}%"
    )

    print(
        f"QQQ:           "
        f"{qqq_return:.2f}%"
    )


    return {
        "label":
            label,

        "return":
            total_return,

        "drawdown":
            max_drawdown,

        "trades":
            len(trades),

        "win_rate":
            win_rate,

        "spy":
            spy_return,

        "qqq":
            qqq_return
    }


# ============================================================
# RUN WINDOWS
# ============================================================

results = []


for (
    label,
    start_date,
    end_date
) in WINDOWS:

    result = test_window(
        label,
        start_date,
        end_date
    )


    if result is not None:

        results.append(
            result
        )


# ============================================================
# REPORT
# ============================================================

print("\n\n")
print("=" * 92)
print("               ROLLING WALK-FORWARD REPORT")
print("=" * 92)


print(
    f"{'WINDOW':<12}"
    f"{'AI':>11}"
    f"{'SPY':>11}"
    f"{'QQQ':>11}"
    f"{'DD':>11}"
    f"{'TRADES':>10}"
    f"{'WIN%':>10}"
)


print("-" * 92)


for result in results:

    print(
        f"{result['label']:<12}"

        f"{result['return']:>10.2f}%"

        f"{result['spy']:>10.2f}%"

        f"{result['qqq']:>10.2f}%"

        f"{result['drawdown']:>10.2f}%"

        f"{result['trades']:>10}"

        f"{result['win_rate']:>9.2f}%"
    )


print("=" * 92)


# ============================================================
# SUMMARY STATS
# ============================================================

if results:

    profitable_windows = sum(
        1
        for r in results
        if r[
            "return"
        ] > 0
    )


    beat_spy = sum(
        1
        for r in results
        if r[
            "return"
        ] > r[
            "spy"
        ]
    )


    beat_qqq = sum(
        1
        for r in results
        if r[
            "return"
        ] > r[
            "qqq"
        ]
    )


    average_return = np.mean(
        [
            r[
                "return"
            ]
            for r in results
        ]
    )


    worst_window = min(
        results,
        key=lambda x:
            x[
                "return"
            ]
    )


    print(
        f"\nProfitable windows: "
        f"{profitable_windows}/"
        f"{len(results)}"
    )


    print(
        f"Beat SPY: "
        f"{beat_spy}/"
        f"{len(results)}"
    )


    print(
        f"Beat QQQ: "
        f"{beat_qqq}/"
        f"{len(results)}"
    )


    print(
        f"Average window return: "
        f"{average_return:.2f}%"
    )


    print(
        f"Worst window: "
        f"{worst_window['label']} "
        f"({worst_window['return']:.2f}%)"
    )