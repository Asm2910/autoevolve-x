from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "artifacts"
    / "xgboost_classifier.joblib"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "xgb_test_predictions.parquet"
)

FROZEN_THRESHOLD = 0.45

TARGET_COLUMN = "risk_label"

ID_COLUMN = "subscriber_id"

CATEGORICAL_COLUMNS = [
    "customer_segment",
    "plan_type",
    "region",
    "device_type",
    "age_band",
    "preferred_language",
    "account_status",
]


def load_test_data() -> pd.DataFrame:
    """Load and validate the test dataset."""

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}"
        )

    test = pd.read_parquet(TEST_PATH)

    if test.empty:
        raise ValueError(
            "Test dataset is empty."
        )

    required_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
    }

    missing_columns = required_columns.difference(
        test.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if test[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate subscribers found in test data."
        )

    if test[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Missing target values found in test data."
        )

    return test


def prepare_features(
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare test features using training-time encoding."""

    features = test.drop(
        columns=[
            ID_COLUMN,
            TARGET_COLUMN,
        ]
    ).copy()

    missing_categorical = [
        column
        for column in CATEGORICAL_COLUMNS
        if column not in features.columns
    ]

    if missing_categorical:
        raise ValueError(
            "Missing categorical columns: "
            f"{missing_categorical}"
        )

    features = pd.get_dummies(
        features,
        columns=CATEGORICAL_COLUMNS,
        drop_first=False,
        dtype=float,
    )

    if features.isna().any().any():
        raise ValueError(
            "Missing values found in test features."
        )

    return features


def align_features_with_model(
    features: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """Align test features with the features used during training."""

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Loaded model does not contain "
            "feature_names_in_."
        )

    model_features = list(
        model.feature_names_in_
    )

    missing_features = [
        column
        for column in model_features
        if column not in features.columns
    ]

    extra_features = [
        column
        for column in features.columns
        if column not in model_features
    ]

    if missing_features:
        for column in missing_features:
            features[column] = 0.0

    if extra_features:
        features = features.drop(
            columns=extra_features
        )

    features = features[
        model_features
    ]

    return features


def main() -> None:
    print("Loading test dataset...")

    test = load_test_data()

    print(
        f"Test shape: {test.shape}"
    )

    print("Input validation passed.")

    print("\nLoading XGBoost model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    print("XGBoost model loaded.")

    features = prepare_features(
        test
    )

    features = align_features_with_model(
        features,
        model,
    )

    target = test[
        TARGET_COLUMN
    ]

    print(
        f"Test features: {features.shape}"
    )

    print(
        f"Positive rate: {target.mean():.4f}"
    )

    probabilities = model.predict_proba(
        features
    )[:, 1]

    predictions = (
        probabilities >= FROZEN_THRESHOLD
    ).astype(int)

    precision = precision_score(
        target,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        target,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        target,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        target,
        probabilities,
    )

    pr_auc = average_precision_score(
        target,
        probabilities,
    )

    tn, fp, fn, tp = confusion_matrix(
        target,
        predictions,
        labels=[0, 1],
    ).ravel()

    print(
        f"\nEvaluating at frozen threshold: "
        f"{FROZEN_THRESHOLD:.2f}"
    )

    print("\nTest metrics:")
    print(
        f"  roc_auc      {roc_auc:.4f}"
    )
    print(
        f"  pr_auc       {pr_auc:.4f}"
    )
    print(
        f"  precision    {precision:.4f}"
    )
    print(
        f"  recall       {recall:.4f}"
    )
    print(
        f"  f1           {f1:.4f}"
    )

    print("\nConfusion matrix:")
    print(
        f"  true_negatives   {tn}"
    )
    print(
        f"  false_positives  {fp}"
    )
    print(
        f"  false_negatives  {fn}"
    )
    print(
        f"  true_positives   {tp}"
    )

    output = pd.DataFrame(
        {
            ID_COLUMN: test[
                ID_COLUMN
            ],
            TARGET_COLUMN: target,
            "predicted_probability": probabilities,
            "predicted_label": predictions,
        }
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSaved XGBoost test predictions to:"
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()