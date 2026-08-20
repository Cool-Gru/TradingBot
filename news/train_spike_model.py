import os

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from sklearn.impute import (
    SimpleImputer
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = (
    "news/spike_dataset_v5.csv"
)

MODEL_PATH = (
    "news/spike_model_v5.joblib"
)

OOS_OUTPUT_PATH = (
    "news/spike_oos_predictions_v5.csv"
)


TARGET = "spike_5pct"

INITIAL_TRAIN_FRACTION = 0.50

N_FOLDS = 3

PURGE_DAYS = 10


# ============================================================
# LOAD
# ============================================================

print(
    "\n=========================================="
)

print(
    "       V5 NEWS + MARKET TRAINER"
)

print(
    "=========================================="
)


if not os.path.exists(
    DATASET_PATH
):

    raise FileNotFoundError(
        f"Missing {DATASET_PATH}"
    )


df = pd.read_csv(
    DATASET_PATH
)


print(
    f"\nRows loaded: "
    f"{len(df):,}"
)


# ============================================================
# CLEAN
# ============================================================

df[
    "event_time"
] = pd.to_datetime(
    df[
        "event_time"
    ],
    utc=True
)


df = (
    df
    .sort_values(
        "event_time"
    )
    .reset_index(
        drop=True
    )
)


df[
    "headline"
] = (
    df[
        "headline"
    ]
    .fillna("")
    .astype(str)
)


df[
    "summary"
] = (
    df[
        "summary"
    ]
    .fillna("")
    .astype(str)
)


df[
    "text"
] = (
    df[
        "headline"
    ]
    + " "
    + df[
        "summary"
    ]
)


df[
    "event_type"
] = (
    df[
        "event_type"
    ]
    .fillna(
        "OTHER"
    )
    .astype(str)
)


# ============================================================
# NUMERIC FEATURES
# ============================================================

NUMERIC_COLUMNS = [

    # NEWS
    "event_score",
    "sentiment",
    "future_score",
    "importance",

    # STOCK
    "stock_history_days",

    "stock_return_1d",
    "stock_return_5d",
    "stock_return_20d",
    "stock_return_60d",

    "stock_volatility_20",
    "stock_volume_ratio_20",

    "stock_rsi_14",

    "stock_price_vs_sma20",
    "stock_price_vs_sma50",
    "stock_price_vs_sma200",

    "stock_gap_1d",
    "stock_range_1d",

    # SPY
    "spy_return_5d",
    "spy_return_20d",
    "spy_volatility_20",
    "spy_price_vs_sma50",
    "spy_price_vs_sma200",

    # QQQ
    "qqq_return_5d",
    "qqq_return_20d",
    "qqq_volatility_20",
    "qqq_price_vs_sma50",
    "qqq_price_vs_sma200",

    # RELATIVE STRENGTH
    "relative_spy_5d",
    "relative_spy_20d",

    "relative_qqq_5d",
    "relative_qqq_20d",
]


for column in NUMERIC_COLUMNS:

    if column not in df.columns:

        raise RuntimeError(
            f"Dataset is missing "
            f"feature: {column}"
        )

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df[
    TARGET
] = pd.to_numeric(
    df[
        TARGET
    ],
    errors="coerce"
)


df = (
    df
    .dropna(
        subset=[
            TARGET
        ]
    )
    .reset_index(
        drop=True
    )
)


df[
    TARGET
] = (
    df[
        TARGET
    ]
    .astype(int)
)


# ============================================================
# FEATURE LIST
# ============================================================

FEATURE_COLUMNS = [

    "text",
    "event_type",

    *NUMERIC_COLUMNS,
]


# ============================================================
# MODEL FACTORY
# ============================================================

def build_model():

    numeric_pipeline = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )
            ),

            (
                "scaler",

                StandardScaler()
            ),
        ]
    )


    preprocessor = ColumnTransformer(

        transformers=[

            # ------------------------------------------------
            # NLP
            # ------------------------------------------------

            (
                "text",

                TfidfVectorizer(

                    lowercase=True,

                    ngram_range=(
                        1,
                        2
                    ),

                    min_df=5,

                    max_df=0.98,

                    max_features=30000,

                    sublinear_tf=True,
                ),

                "text",
            ),


            # ------------------------------------------------
            # EVENT TYPE
            # ------------------------------------------------

            (
                "event",

                OneHotEncoder(
                    handle_unknown="ignore"
                ),

                [
                    "event_type"
                ],
            ),


            # ------------------------------------------------
            # STOCK / MARKET FEATURES
            # ------------------------------------------------

            (
                "numeric",

                numeric_pipeline,

                NUMERIC_COLUMNS,
            ),
        ]
    )


    classifier = LogisticRegression(

        max_iter=3000,

        C=1.0,

        solver="liblinear",

        random_state=42,
    )


    return Pipeline(

        steps=[

            (
                "features",
                preprocessor
            ),

            (
                "model",
                classifier
            ),
        ]
    )


# ============================================================
# SAFE METRICS
# ============================================================

def safe_roc_auc(
    y_true,
    probabilities
):

    if len(
        np.unique(
            y_true
        )
    ) < 2:

        return float(
            "nan"
        )

    return roc_auc_score(
        y_true,
        probabilities
    )


def safe_pr_auc(
    y_true,
    probabilities
):

    if len(
        np.unique(
            y_true
        )
    ) < 2:

        return float(
            "nan"
        )

    return average_precision_score(
        y_true,
        probabilities
    )


# ============================================================
# DATA RANGE
# ============================================================

dataset_start = (
    df[
        "event_time"
    ].min()
)


dataset_end = (
    df[
        "event_time"
    ].max()
)


duration = (
    dataset_end
    - dataset_start
)


initial_test_start = (
    dataset_start
    + duration
    * INITIAL_TRAIN_FRACTION
)


remaining_duration = (
    dataset_end
    - initial_test_start
)


fold_duration = (
    remaining_duration
    / N_FOLDS
)


print(
    "\n------------------------------------------"
)

print(
    "DATA RANGE"
)

print(
    "------------------------------------------"
)


print(
    f"Start: "
    f"{dataset_start}"
)


print(
    f"End:   "
    f"{dataset_end}"
)


print(
    f"\nPurging "
    f"{PURGE_DAYS} calendar days "
    f"before each test window."
)


# ============================================================
# WALK-FORWARD
# ============================================================

all_oos_predictions = []


for fold_number in range(
    N_FOLDS
):

    test_start = (
        initial_test_start
        + fold_duration
        * fold_number
    )


    if (
        fold_number
        == N_FOLDS - 1
    ):

        test_end = (
            dataset_end
            + pd.Timedelta(
                seconds=1
            )
        )

    else:

        test_end = (
            initial_test_start
            + fold_duration
            * (
                fold_number
                + 1
            )
        )


    train_end = (
        test_start
        - pd.Timedelta(
            days=PURGE_DAYS
        )
    )


    train_df = (
        df[
            df[
                "event_time"
            ] < train_end
        ]
        .copy()
    )


    test_df = (
        df[
            (
                df[
                    "event_time"
                ]
                >= test_start
            )
            &
            (
                df[
                    "event_time"
                ]
                < test_end
            )
        ]
        .copy()
    )


    if (
        len(train_df) < 500
        or
        len(test_df) < 100
    ):

        print(
            f"\nFold {fold_number + 1} "
            f"skipped."
        )

        continue


    print(
        "\n=========================================="
    )

    print(
        f"               FOLD "
        f"{fold_number + 1}"
    )

    print(
        "=========================================="
    )


    print(
        "\nTraining:"
    )

    print(
        train_df[
            "event_time"
        ].min()
    )

    print(
        "→"
    )

    print(
        train_df[
            "event_time"
        ].max()
    )


    print(
        "\nTesting:"
    )

    print(
        test_start
    )

    print(
        "→"
    )

    print(
        test_end
    )


    print(
        f"\nTrain rows: "
        f"{len(train_df):,}"
    )

    print(
        f"Test rows:  "
        f"{len(test_df):,}"
    )


    train_rate = (
        train_df[
            TARGET
        ].mean()
    )


    test_rate = (
        test_df[
            TARGET
        ].mean()
    )


    print(
        f"\nTraining baseline: "
        f"{train_rate:.2%}"
    )


    print(
        f"Testing baseline:  "
        f"{test_rate:.2%}"
    )


    X_train = (
        train_df[
            FEATURE_COLUMNS
        ]
    )


    y_train = (
        train_df[
            TARGET
        ]
    )


    X_test = (
        test_df[
            FEATURE_COLUMNS
        ]
    )


    y_test = (
        test_df[
            TARGET
        ]
    )


    model = build_model()


    print(
        "\nTraining model..."
    )


    model.fit(
        X_train,
        y_train
    )


    probabilities = (
        model
        .predict_proba(
            X_test
        )[:, 1]
    )


    predictions = (
        probabilities
        >= 0.50
    ).astype(int)


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


    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    roc_auc = safe_roc_auc(
        y_test,
        probabilities
    )


    pr_auc = safe_pr_auc(
        y_test,
        probabilities
    )


    print(
        "\nFold results:"
    )


    print(
        f"  Accuracy:  "
        f"{accuracy:.2%}"
    )


    print(
        f"  Precision: "
        f"{precision:.2%}"
    )


    print(
        f"  Recall:    "
        f"{recall:.2%}"
    )


    print(
        f"  F1:        "
        f"{f1:.3f}"
    )


    print(
        f"  ROC AUC:   "
        f"{roc_auc:.3f}"
    )


    print(
        f"  PR AUC:    "
        f"{pr_auc:.3f}"
    )


    result_df = pd.DataFrame({

        "event_time":
            test_df[
                "event_time"
            ].to_numpy(),

        "symbol":
            test_df[
                "symbol"
            ].to_numpy(),

        "headline":
            test_df[
                "headline"
            ].to_numpy(),

        "actual":
            y_test.to_numpy(),

        "probability":
            probabilities,

        "fold":
            fold_number + 1,

        "fold_baseline":
            test_rate,
    })


    all_oos_predictions.append(
        result_df
    )


# ============================================================
# COMBINE OOS RESULTS
# ============================================================

if not all_oos_predictions:

    raise RuntimeError(
        "No valid walk-forward folds."
    )


oos = pd.concat(
    all_oos_predictions,
    ignore_index=True
)


oos = (
    oos
    .sort_values(
        "event_time"
    )
    .reset_index(
        drop=True
    )
)


oos.to_csv(
    OOS_OUTPUT_PATH,
    index=False
)


actual = (
    oos[
        "actual"
    ].to_numpy()
)


probabilities = (
    oos[
        "probability"
    ].to_numpy()
)


baseline = float(
    actual.mean()
)


print(
    "\n=========================================="
)

print(
    "      COMBINED OUT-OF-SAMPLE RESULTS"
)

print(
    "=========================================="
)


print(
    f"\nOOS examples: "
    f"{len(oos):,}"
)


print(
    f"Baseline spike rate: "
    f"{baseline:.2%}"
)


print(
    f"ROC AUC: "
    f"{safe_roc_auc(actual, probabilities):.3f}"
)


print(
    f"PR AUC: "
    f"{safe_pr_auc(actual, probabilities):.3f}"
)


# ============================================================
# THRESHOLDS
# ============================================================

print(
    "\n=========================================="
)

print(
    "        CONFIDENCE THRESHOLDS"
)

print(
    "=========================================="
)


THRESHOLDS = [

    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
]


threshold_summary = []


for threshold in THRESHOLDS:

    mask = (
        probabilities
        >= threshold
    )


    count = int(
        mask.sum()
    )


    if count == 0:

        print(
            f"\n>= {threshold:.0%}: "
            f"0 signals"
        )

        continue


    selected_actual = (
        actual[
            mask
        ]
    )


    spikes = int(
        selected_actual.sum()
    )


    precision = (
        spikes
        / count
    )


    total_spikes = int(
        actual.sum()
    )


    recall = (
        spikes
        / total_spikes
        if total_spikes > 0
        else 0
    )


    average_probability = float(
        probabilities[
            mask
        ].mean()
    )


    lift = (
        precision
        / baseline
        if baseline > 0
        else 0
    )


    print(
        f"\n>= {threshold:.0%}"
    )


    print(
        f"  Signals:    "
        f"{count:,}"
    )


    print(
        f"  Spikes:     "
        f"{spikes:,}"
    )


    print(
        f"  Precision:  "
        f"{precision:.2%}"
    )


    print(
        f"  Recall:     "
        f"{recall:.2%}"
    )


    print(
        f"  Lift:       "
        f"{lift:.2f}x baseline"
    )


    print(
        f"  Avg score:  "
        f"{average_probability:.2%}"
    )


    threshold_summary.append({

        "threshold":
            threshold,

        "signals":
            count,

        "precision":
            precision,

        "recall":
            recall,

        "lift":
            lift,
    })


# ============================================================
# TOP RANKED SIGNALS
# ============================================================

print(
    "\n=========================================="
)

print(
    "       TOP CONFIDENCE PERFORMANCE"
)

print(
    "=========================================="
)


ranked = (
    oos
    .sort_values(
        "probability",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


for top_n in [

    25,
    50,
    100,
    250,
    500,
    1000,
]:

    if len(
        ranked
    ) < top_n:
        continue


    sample = ranked.head(
        top_n
    )


    precision = float(
        sample[
            "actual"
        ].mean()
    )


    lift = (
        precision
        / baseline
        if baseline > 0
        else 0
    )


    print(
        f"\nTop {top_n:>4} signals:"
    )


    print(
        f"  Spike rate: "
        f"{precision:.2%}"
    )


    print(
        f"  Lift:       "
        f"{lift:.2f}x"
    )


# ============================================================
# 60%+ BY FOLD
# ============================================================

print(
    "\n=========================================="
)

print(
    "       60%+ SIGNALS BY FOLD"
)

print(
    "=========================================="
)


for fold in sorted(
    oos[
        "fold"
    ].unique()
):

    fold_df = (
        oos[
            oos[
                "fold"
            ] == fold
        ]
    )


    fold_baseline = float(
        fold_df[
            "actual"
        ].mean()
    )


    selected = (
        fold_df[
            fold_df[
                "probability"
            ] >= 0.60
        ]
    )


    print(
        f"\nFold {fold}:"
    )


    print(
        f"  Baseline:  "
        f"{fold_baseline:.2%}"
    )


    print(
        f"  Signals:   "
        f"{len(selected):,}"
    )


    if len(
        selected
    ) > 0:

        print(
            f"  Precision: "
            f"{selected['actual'].mean():.2%}"
        )

    else:

        print(
            "  Precision: N/A"
        )


# ============================================================
# 65%+ BY FOLD
# ============================================================

print(
    "\n=========================================="
)

print(
    "       65%+ SIGNALS BY FOLD"
)

print(
    "=========================================="
)


for fold in sorted(
    oos[
        "fold"
    ].unique()
):

    fold_df = (
        oos[
            oos[
                "fold"
            ] == fold
        ]
    )


    fold_baseline = float(
        fold_df[
            "actual"
        ].mean()
    )


    selected = (
        fold_df[
            fold_df[
                "probability"
            ] >= 0.65
        ]
    )


    print(
        f"\nFold {fold}:"
    )


    print(
        f"  Baseline:  "
        f"{fold_baseline:.2%}"
    )


    print(
        f"  Signals:   "
        f"{len(selected):,}"
    )


    if len(
        selected
    ) > 0:

        print(
            f"  Precision: "
            f"{selected['actual'].mean():.2%}"
        )

    else:

        print(
            "  Precision: N/A"
        )


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print(
    "\n=========================================="
)

print(
    "          TRAINING FINAL V5 MODEL"
)

print(
    "=========================================="
)


final_model = build_model()


final_model.fit(
    df[
        FEATURE_COLUMNS
    ],
    df[
        TARGET
    ]
)


# ============================================================
# SAVE
# ============================================================

bundle = {

    "model":
        final_model,

    "features":
        FEATURE_COLUMNS,

    "numeric_features":
        NUMERIC_COLUMNS,

    "target":
        TARGET,

    "purge_days":
        PURGE_DAYS,

    "validation_folds":
        N_FOLDS,

    "dataset_start":
        str(
            dataset_start
        ),

    "dataset_end":
        str(
            dataset_end
        ),

    "oos_baseline":
        baseline,

    "oos_roc_auc":
        float(
            safe_roc_auc(
                actual,
                probabilities
            )
        ),

    "oos_pr_auc":
        float(
            safe_pr_auc(
                actual,
                probabilities
            )
        ),

    "threshold_results":
        threshold_summary,
}


joblib.dump(
    bundle,
    MODEL_PATH
)


print(
    "\n=========================================="
)

print(
    "              V5 MODEL SAVED"
)

print(
    "=========================================="
)


print(
    f"\n{MODEL_PATH}"
)


print(
    "\nOOS predictions:"
)


print(
    OOS_OUTPUT_PATH
)