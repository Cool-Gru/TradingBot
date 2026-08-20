from datetime import datetime, timezone
import joblib

import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    roc_auc_score
)


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
    "SPY",
    "QQQ"
]

DATA_START = datetime(
    2018, 1, 1,
    tzinfo=timezone.utc
)

DATA_END = datetime.now(
    timezone.utc
)

TRAIN_END = pd.Timestamp(
    "2024-12-31",
    tz="UTC"
)


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
    "daily_range"
]


# ==========================================
# BUILD DATASET
# ==========================================

all_data = []

print("\nDownloading and building dataset...\n")

for symbol in SYMBOLS:

    print(f"Processing {symbol}...")

    df = get_daily_data(
        symbol,
        DATA_START,
        DATA_END
    )

    df = build_features(df)

    df["symbol_name"] = symbol

    all_data.append(df)


dataset = pd.concat(
    all_data,
    ignore_index=True
)

dataset = dataset.sort_values(
    "timestamp"
).reset_index(drop=True)


print(
    f"\nTotal samples: {len(dataset)}"
)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

train_df = dataset[
    dataset["timestamp"] <= TRAIN_END
].copy()

test_df = dataset[
    dataset["timestamp"] > TRAIN_END
].copy()


X_train = train_df[FEATURES]
y_train = train_df["target"]

X_test = test_df[FEATURES]
y_test = test_df["target"]


print("\n===================================")
print("          DATASET SPLIT")
print("===================================")

print(
    f"Training samples: "
    f"{len(X_train)}"
)

print(
    f"Testing samples:  "
    f"{len(X_test)}"
)


# ==========================================
# CLASS DISTRIBUTION
# ==========================================

print("\n===================================")
print("       CLASS DISTRIBUTION")
print("===================================")

train_positive_rate = (
    y_train.mean() * 100
)

test_positive_rate = (
    y_test.mean() * 100
)

print(
    f"Training positive rate: "
    f"{train_positive_rate:.2f}%"
)

print(
    f"Testing positive rate:  "
    f"{test_positive_rate:.2f}%"
)


# ==========================================
# BASELINE
# ==========================================

most_common_class = (
    y_train
    .value_counts()
    .idxmax()
)

baseline_predictions = [
    most_common_class
] * len(y_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)


# ==========================================
# MODEL
# ==========================================

model = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=200,
    max_leaf_nodes=15,
    min_samples_leaf=30,
    l2_regularization=1.0,
    random_state=42
)


print("\nTraining model...")

model.fit(
    X_train,
    y_train
)
joblib.dump(
    model,
    "ml/trading_model.joblib"
)

print(
    "Model saved to "
    "ml/trading_model.joblib"
)

print("Training complete.")


# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ==========================================
# METRICS
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


print("\n===================================")
print("          MODEL RESULTS")
print("===================================")

print(
    f"Baseline Accuracy: "
    f"{baseline_accuracy * 100:.2f}%"
)

print(
    f"Model Accuracy:    "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision:         "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall:            "
    f"{recall * 100:.2f}%"
)

print(
    f"ROC AUC:           "
    f"{roc_auc:.3f}"
)


print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions,
        digits=3
    )
)


# ==========================================
# RESULTS DATAFRAME
# ==========================================

results = test_df[
    [
        "timestamp",
        "symbol_name",
        "close",
        "future_return_5d",
        "target"
    ]
].copy()

results["prob_up"] = probabilities

results["prediction"] = predictions


# ==========================================
# PROBABILITY ANALYSIS
# ==========================================

print("\n===================================")
print("       PROBABILITY ANALYSIS")
print("===================================")


thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]


for threshold in thresholds:

    signals = results[
        results["prob_up"] >= threshold
    ]

    print(
        f"\n--- P >= {threshold:.2f} ---"
    )

    if len(signals) == 0:

        print("Signals: 0")

        continue


    success_rate = (
        signals["target"].mean()
        * 100
    )

    avg_return = (
        signals[
            "future_return_5d"
        ].mean()
        * 100
    )


    median_return = (
        signals[
            "future_return_5d"
        ].median()
        * 100
    )


    print(
        f"Signals:        "
        f"{len(signals)}"
    )

    print(
        f"Success Rate:   "
        f"{success_rate:.2f}%"
    )

    print(
        f"Avg 5D Return:  "
        f"{avg_return:.2f}%"
    )

    print(
        f"Median Return:  "
        f"{median_return:.2f}%"
    )


# ==========================================
# TOP MODEL SIGNALS
# ==========================================

print("\n===================================")
print("          TOP SIGNALS")
print("===================================")

top_signals = (
    results
    .sort_values(
        "prob_up",
        ascending=False
    )
    .head(20)
)


print(
    top_signals[
        [
            "timestamp",
            "symbol_name",
            "close",
            "prob_up",
            "future_return_5d",
            "target"
        ]
    ]
    .to_string(
        index=False
    )
)