from datetime import datetime, timezone

import joblib
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


DATA_START = datetime(
    2010,
    1,
    1,
    tzinfo=timezone.utc
)

DATA_END = datetime.now(
    timezone.utc
)


# ============================================================
# DOWNLOAD MARKET DATA
# ============================================================

print("\n============================================")
print("             TRAINING V3 MODEL")
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


# ============================================================
# BUILD TRAINING DATA
# ============================================================

frames = []


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


    combined[
        "symbol_name"
    ] = symbol


    frames.append(
        combined
    )


dataset = pd.concat(
    frames,
    ignore_index=True
)


dataset = dataset.sort_values(
    "timestamp"
).reset_index(drop=True)


print(
    f"\nTotal training samples: "
    f"{len(dataset)}"
)

print(
    f"Features: "
    f"{len(FEATURES)}"
)


# ============================================================
# TRAIN
# ============================================================

X = dataset[
    FEATURES
]

y = dataset[
    "target"
]


model = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=200,
    max_leaf_nodes=15,
    min_samples_leaf=30,
    l2_regularization=1.0,
    random_state=42
)


print(
    "\nTraining V3 model..."
)

model.fit(
    X,
    y
)

print(
    "Training complete."
)


# ============================================================
# SAVE MODEL + METADATA
# ============================================================

bundle = {

    "model":
        model,

    "features":
        FEATURES,

    "symbols":
        SYMBOLS,

    "buy_threshold":
        0.65,

    "hold_days":
        5,

    "requires_regime_filter":
        True,

    "trained_through":
        str(
            dataset[
                "timestamp"
            ].max()
        )
}


joblib.dump(
    bundle,
    "ml/trading_model_v3.joblib"
)


print(
    "\nSaved:"
)

print(
    "ml/trading_model_v3.joblib"
)

print(
    "\nModel features:"
)

for feature in FEATURES:

    print(
        f" - {feature}"
    )