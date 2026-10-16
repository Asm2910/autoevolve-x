from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.tree_model import (
    predict_proba,
    save_model,
    train_xgb_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.parquet"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "artifacts"
    / "xgboost_classifier.joblib"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "xgb_validation_predictions.parquet"
)


def validate_input(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Validate train and validation datasets."""

    required_columns = {
        "subscriber_id",
        "risk_label",
    }

    for name, dataframe in (
        ("train", train),
        ("validation", validation),
    ):
        missing_columns = (
            required_columns
            - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{name} missing columns: "
                f"{sorted(missing_columns)}"
            )

        if dataframe[
            "subscriber_id"
        ].duplicated().any():
            raise ValueError(
                f"{name} contains duplicate "
                "subscriber IDs."
            )

        if dataframe[
            "risk_label"
        ].isna().any():
            raise ValueError(
                f"{name} contains missing labels."
            )

        invalid_labels = set(
            dataframe["risk_label"].unique()
        ) - {0, 1}

        if invalid_labels:
            raise ValueError(
                f"{name} contains invalid labels: "
                f"{invalid_labels}"
            )


def calculate_metrics(
    y_true: pd.Series,
    probabilities: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate classification metrics."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }


def main() -> None:
    print("Loading datasets...")

    train = pd.read_parquet(
        TRAIN_PATH
    )

    validation = pd.read_parquet(
        VALIDATION_PATH
    )

    print(
        f"Train shape: {train.shape}"
    )

    print(
        f"Validation shape: {validation.shape}"
    )

    validate_input(
        train,
        validation,
    )

    print("Input validation passed.")

    print(
        "\nTraining XGBoost model..."
    )

    artifact = train_xgb_model(
        train
    )

    print(
        "XGBoost model training completed."
    )

    validation_probabilities = predict_proba(
        artifact,
        validation,
    )

    y_validation = validation[
        "risk_label"
    ]

    metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
        threshold=0.5,
    )

    print("\nValidation metrics:")

    for name, value in metrics.items():
        print(
            f"  {name:<12} {value:.4f}"
        )

    validation_predictions = pd.DataFrame(
        {
            "subscriber_id": validation[
                "subscriber_id"
            ],
            "risk_label": y_validation,
            "predicted_probability": (
                validation_probabilities
            ),
            "predicted_label": (
                validation_probabilities >= 0.5
            ).astype(int),
        }
    )

    save_model(
        artifact,
        MODEL_PATH,
    )

    validation_predictions.to_parquet(
        PREDICTION_PATH,
        index=False,
    )

    print(
        "\nSaved model to:"
    )
    print(MODEL_PATH)

    print(
        "\nSaved validation predictions to:"
    )
    print(PREDICTION_PATH)


if __name__ == "__main__":
    main()