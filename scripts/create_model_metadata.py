from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

MODEL_REPORT_PATH = Path(
    "data/processed/model_report.json"
)

MODEL_COMPARISON_PATH = Path(
    "data/processed/model_comparison.parquet"
)

MODEL_PATH = Path(
    "models/artifacts/baseline_logistic_regression.joblib"
)

OUTPUT_PATH = Path(
    "models/artifacts/model_metadata.json"
)

MODEL_NAME = "logistic_regression"
MODEL_VERSION = "1.0.0"
THRESHOLD = 0.90


def load_model_report() -> dict:
    """Load the generated model evaluation report."""
    if not MODEL_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Model report not found: {MODEL_REPORT_PATH}"
        )

    with MODEL_REPORT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_model_comparison() -> pd.DataFrame:
    """Load model comparison results."""
    if not MODEL_COMPARISON_PATH.exists():
        raise FileNotFoundError(
            "Model comparison not found: "
            f"{MODEL_COMPARISON_PATH}"
        )

    return pd.read_parquet(
        MODEL_COMPARISON_PATH
    )


def validate_artifacts() -> None:
    """Validate required model artifacts."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )


def extract_model_metrics(
    report: dict,
    comparison: pd.DataFrame,
) -> dict:
    """Extract metrics for the selected model."""
    model_rows = comparison[
        comparison["model"] == MODEL_NAME
    ]

    if model_rows.empty:
        raise ValueError(
            f"Model not found in comparison: "
            f"{MODEL_NAME}"
        )

    row = model_rows.iloc[0]

    return {
        "roc_auc": float(row["roc_auc"]),
        "pr_auc": float(row["pr_auc"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
        "threshold": float(row["threshold"]),
    }


def build_metadata(
    report: dict,
    comparison: pd.DataFrame,
) -> dict:
    """Build model metadata."""
    metrics = extract_model_metrics(
        report,
        comparison,
    )

    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_type": "logistic_regression",
        "artifact_path": str(MODEL_PATH),
        "threshold": THRESHOLD,
        "metrics": metrics,
        "training_dataset": {
            "path": "data/processed/train.parquet",
            "rows": 14000,
        },
        "validation_dataset": {
            "path": "data/processed/validation.parquet",
            "rows": 3000,
        },
        "test_dataset": {
            "path": "data/processed/test.parquet",
            "rows": 3000,
        },
        "feature_dataset": {
            "path": "data/processed/features.parquet",
            "rows": 20000,
        },
        "prediction_output": {
            "path": "data/processed/predictions.parquet",
        },
        "created_at": datetime.now(
            UTC
        ).isoformat(),
    }


def validate_metadata(
    metadata: dict,
) -> None:
    """Validate model metadata."""
    required_fields = {
        "model_name",
        "model_version",
        "model_type",
        "artifact_path",
        "threshold",
        "metrics",
        "training_dataset",
        "validation_dataset",
        "test_dataset",
        "feature_dataset",
        "prediction_output",
        "created_at",
    }

    missing_fields = (
        required_fields
        - set(metadata)
    )

    if missing_fields:
        raise ValueError(
            "Missing metadata fields: "
            f"{sorted(missing_fields)}"
        )

    if not 0 <= metadata["threshold"] <= 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    required_metrics = {
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "threshold",
    }

    missing_metrics = (
        required_metrics
        - set(metadata["metrics"])
    )

    if missing_metrics:
        raise ValueError(
            "Missing model metrics: "
            f"{sorted(missing_metrics)}"
        )


def save_metadata(
    metadata: dict,
) -> None:
    """Save model metadata as JSON."""
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )


def main() -> None:
    """Create versioned model metadata."""
    print("Creating model metadata...")

    validate_artifacts()

    report = load_model_report()
    comparison = load_model_comparison()

    metadata = build_metadata(
        report,
        comparison,
    )

    validate_metadata(metadata)
    save_metadata(metadata)

    print()
    print("Model metadata created.")
    print(
        f"Model: {metadata['model_name']}"
    )
    print(
        f"Version: {metadata['model_version']}"
    )
    print(
        f"Threshold: {metadata['threshold']:.2f}"
    )
    print(
        f"F1: {metadata['metrics']['f1']:.4f}"
    )
    print(
        f"PR-AUC: {metadata['metrics']['pr_auc']:.4f}"
    )
    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()