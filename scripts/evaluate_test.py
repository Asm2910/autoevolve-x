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
    / "baseline_logistic_regression.joblib"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_test_predictions.parquet"
)

THRESHOLD = 0.90


def validate_test_data(test: pd.DataFrame) -> None:
    """Validate the test dataset before inference."""

    required_columns = {
        "subscriber_id",
        "risk_label",
    }

    missing_columns = (
        required_columns - set(test.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if test["subscriber_id"].duplicated().any():
        raise ValueError(
            "Test dataset contains duplicate subscribers."
        )

    if test["risk_label"].isna().any():
        raise ValueError(
            "Test dataset contains missing labels."
        )

    if test.empty:
        raise ValueError(
            "Test dataset is empty."
        )


def main() -> None:
    """Evaluate the frozen baseline model on the test set."""

    print("Loading test dataset...")

    test = pd.read_parquet(TEST_PATH)

    print(
        f"Test shape: {test.shape}"
    )

    validate_test_data(test)

    print("Input validation passed.")

    print("\nLoading baseline model...")

    model = joblib.load(MODEL_PATH)

    print("Baseline model loaded.")

    feature_columns = [
        column
        for column in test.columns
        if column not in {
            "subscriber_id",
            "risk_label",
        }
    ]

    X_test = test[feature_columns]
    y_test = test["risk_label"]

    print(
        f"Test features: {X_test.shape}"
    )

    print(
        f"Positive rate: {y_test.mean():.4f}"
    )

    print(
        f"\nEvaluating at frozen threshold: "
        f"{THRESHOLD:.2f}"
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

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

    predictions_df = pd.DataFrame(
        {
            "subscriber_id": test[
                "subscriber_id"
            ],
            "risk_label": y_test,
            "risk_probability": probabilities,
            "risk_prediction": predictions,
        }
    )

    predictions_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSaved test predictions to:"
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()