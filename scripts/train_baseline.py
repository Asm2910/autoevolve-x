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

from models.baseline import (
    ID_COLUMN,
    TARGET_COLUMN,
    save_model,
    train_baseline,
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
    / "baseline_logistic_regression.joblib"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_validation_predictions.parquet"
)


def validate_input(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> None:
    """Validate training and validation datasets."""

    required_columns = {
        TARGET_COLUMN,
        ID_COLUMN,
    }

    for name, dataframe in (
        ("train", train_df),
        ("validation", validation_df),
    ):
        missing = required_columns - set(
            dataframe.columns
        )

        if missing:
            raise ValueError(
                f"{name} dataset is missing columns: "
                f"{sorted(missing)}"
            )

        if dataframe[ID_COLUMN].duplicated().any():
            raise ValueError(
                f"{name} contains duplicate subscribers."
            )

    train_features = set(train_df.columns) - {
        TARGET_COLUMN
    }

    validation_features = set(
        validation_df.columns
    ) - {TARGET_COLUMN}

    if train_features != validation_features:
        raise ValueError(
            "Train and validation feature columns "
            "do not match."
        )

    overlap = set(
        train_df[ID_COLUMN]
    ).intersection(
        validation_df[ID_COLUMN]
    )

    if overlap:
        raise ValueError(
            "Subscriber overlap detected between "
            f"train and validation: {len(overlap)}"
        )


def calculate_metrics(
    y_true: pd.Series,
    probabilities,
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

    train_df = pd.read_parquet(
        TRAIN_PATH
    )

    validation_df = pd.read_parquet(
        VALIDATION_PATH
    )

    print(
        f"Train shape: {train_df.shape}"
    )

    print(
        f"Validation shape: "
        f"{validation_df.shape}"
    )

    validate_input(
        train_df,
        validation_df,
    )

    print("Input validation passed.")

    print(
        "\nTraining baseline model..."
    )

    model = train_baseline(
        train_df
    )

    print(
        "Baseline model training completed."
    )

    X_validation = validation_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_validation = validation_df[
        TARGET_COLUMN
    ]

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
    )

    print("\nValidation metrics:")

    for metric, value in metrics.items():
        print(
            f"  {metric:<12} "
            f"{value:.4f}"
        )

    validation_predictions = (
        validation_df[
            [ID_COLUMN, TARGET_COLUMN]
        ]
        .copy()
    )

    validation_predictions[
        "risk_probability"
    ] = validation_probabilities

    validation_predictions[
        "risk_prediction"
    ] = (
        validation_probabilities >= 0.5
    ).astype(int)

    PREDICTION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_predictions.to_parquet(
        PREDICTION_PATH,
        index=False,
    )

    save_model(
        model,
        MODEL_PATH,
    )

    print(
        "\nSaved model to:"
    )

    print(
        f"  {MODEL_PATH}"
    )

    print(
        "\nSaved validation predictions to:"
    )

    print(
        f"  {PREDICTION_PATH}"
    )


if __name__ == "__main__":
    main()