from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_validation_predictions.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_threshold_analysis.parquet"
)

TARGET_COLUMN = "risk_label"
PROBABILITY_COLUMN = "risk_probability"

THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


def validate_predictions(
    predictions: pd.DataFrame,
) -> None:
    """Validate baseline prediction dataset."""

    required_columns = {
        TARGET_COLUMN,
        PROBABILITY_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if predictions.empty:
        raise ValueError(
            "Prediction dataset is empty."
        )

    if predictions[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    if predictions[
        PROBABILITY_COLUMN
    ].isna().any():
        raise ValueError(
            "Risk probability contains "
            "missing values."
        )

    probabilities = predictions[
        PROBABILITY_COLUMN
    ]

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(
            "Risk probabilities must be "
            "between 0 and 1."
        )


def evaluate_threshold(
    y_true: pd.Series,
    probabilities: pd.Series,
    threshold: float,
) -> dict[str, float | int]:
    """Evaluate classification performance at a threshold."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    predicted_positive_rate = (
        predictions.mean()
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": (
            predicted_positive_rate
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def main() -> None:
    print(
        "Loading baseline validation "
        "predictions..."
    )

    predictions = pd.read_parquet(
        PREDICTION_PATH
    )

    print(
        f"Loaded predictions: "
        f"{predictions.shape}"
    )

    validate_predictions(
        predictions
    )

    print(
        "Prediction validation passed."
    )

    y_true = predictions[
        TARGET_COLUMN
    ]

    probabilities = predictions[
        PROBABILITY_COLUMN
    ]

    results = [
        evaluate_threshold(
            y_true=y_true,
            probabilities=probabilities,
            threshold=threshold,
        )
        for threshold in THRESHOLDS
    ]

    threshold_results = pd.DataFrame(
        results
    )

    threshold_results = (
        threshold_results.sort_values(
            "threshold"
        )
        .reset_index(drop=True)
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
        "\nThreshold analysis:"
    )

    print(
        threshold_results.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
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

    print(
        "\nBest F1 threshold:"
    )

    print(
        f"  threshold="
        f"{best_f1['threshold']:.2f}"
    )

    print(
        f"  precision="
        f"{best_f1['precision']:.4f}"
    )

    print(
        f"  recall="
        f"{best_f1['recall']:.4f}"
    )

    print(
        f"  f1="
        f"{best_f1['f1']:.4f}"
    )

    print(
        "\nBest precision threshold:"
    )

    print(
        f"  threshold="
        f"{best_precision['threshold']:.2f}"
    )

    print(
        f"  precision="
        f"{best_precision['precision']:.4f}"
    )

    print(
        "\nBest recall threshold:"
    )

    print(
        f"  threshold="
        f"{best_recall['threshold']:.2f}"
    )

    print(
        f"  recall="
        f"{best_recall['recall']:.4f}"
    )

    print(
        "\nSaved threshold analysis to:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()