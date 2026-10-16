from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "xgb_validation_predictions.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "xgb_threshold_analysis.parquet"
)

THRESHOLDS = np.arange(
    0.10,
    0.96,
    0.05,
)


def load_predictions() -> pd.DataFrame:
    """Load and validate XGBoost validation predictions."""

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}"
        )

    predictions = pd.read_parquet(
        PREDICTIONS_PATH
    )

    required_columns = {
        "subscriber_id",
        "risk_label",
        "predicted_probability",
    }

    missing_columns = required_columns.difference(
        predictions.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if predictions.empty:
        raise ValueError(
            "Validation predictions are empty."
        )

    if predictions["subscriber_id"].duplicated().any():
        raise ValueError(
            "Duplicate subscribers found in predictions."
        )

    if predictions["risk_label"].isna().any():
        raise ValueError(
            "Missing target values found."
        )

    if predictions[
        "predicted_probability"
    ].isna().any():
        raise ValueError(
            "Missing prediction probabilities found."
        )

    probabilities = predictions[
        "predicted_probability"
    ]

    if not probabilities.between(0, 1).all():
        raise ValueError(
            "Prediction probabilities must be "
            "between 0 and 1."
        )

    return predictions


def evaluate_threshold(
    predictions: pd.DataFrame,
    threshold: float,
) -> dict:
    """Evaluate classification performance at one threshold."""

    y_true = predictions[
        "risk_label"
    ].to_numpy()

    y_probability = predictions[
        "predicted_probability"
    ].to_numpy()

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": y_pred.mean(),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


def main() -> None:
    print(
        "Loading XGBoost validation predictions..."
    )

    predictions = load_predictions()

    print(
        f"Loaded predictions: {predictions.shape}"
    )

    print(
        "Prediction validation passed."
    )

    results = [
        evaluate_threshold(
            predictions,
            float(threshold),
        )
        for threshold in THRESHOLDS
    ]

    threshold_results = pd.DataFrame(
        results
    )

    print("\nXGBoost threshold analysis:")
    print(
        threshold_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    best_f1 = threshold_results.loc[
        threshold_results["f1"].idxmax()
    ]

    best_precision = threshold_results.loc[
        threshold_results["precision"].idxmax()
    ]

    best_recall = threshold_results.loc[
        threshold_results["recall"].idxmax()
    ]

    print("\nBest F1 threshold:")
    print(
        f"  threshold={best_f1['threshold']:.2f}"
    )
    print(
        f"  precision={best_f1['precision']:.4f}"
    )
    print(
        f"  recall={best_f1['recall']:.4f}"
    )
    print(
        f"  f1={best_f1['f1']:.4f}"
    )

    print("\nBest precision threshold:")
    print(
        f"  threshold={best_precision['threshold']:.2f}"
    )
    print(
        f"  precision={best_precision['precision']:.4f}"
    )

    print("\nBest recall threshold:")
    print(
        f"  threshold={best_recall['threshold']:.2f}"
    )
    print(
        f"  recall={best_recall['recall']:.4f}"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_results.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSaved XGBoost threshold analysis to:"
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()