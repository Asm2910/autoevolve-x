from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATA_PATH = Path("data/processed/features.parquet")
MODEL_DIR = Path("models/artifacts")
OUTPUT_DIR = Path("data/processed")

LOGISTIC_MODEL_PATH = MODEL_DIR / "baseline_logistic_regression.joblib"
XGB_MODEL_PATH = MODEL_DIR / "xgboost_classifier.joblib"

LOGISTIC_OUTPUT_PATH = OUTPUT_DIR / "logistic_feature_importance.parquet"
XGB_OUTPUT_PATH = OUTPUT_DIR / "xgb_feature_importance.parquet"

TARGET_COLUMN = "risk_label"
ID_COLUMN = "subscriber_id"


def load_feature_dataset() -> pd.DataFrame:
    """Load and validate the complete feature dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    required_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Feature dataset is empty.")

    print(f"Feature dataset loaded: {df.shape}")

    return df


def prepare_model_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Remove identifier and target columns."""

    model_features = df.drop(
        columns=[
            ID_COLUMN,
            TARGET_COLUMN,
        ],
        errors="ignore",
    ).copy()

    if model_features.empty:
        raise ValueError("No model features available.")

    print(
        f"Original model features: {model_features.shape[1]}"
    )

    return model_features


def load_models() -> tuple[object, object]:
    """Load trained Logistic Regression and XGBoost models."""

    if not LOGISTIC_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Logistic model not found: {LOGISTIC_MODEL_PATH}"
        )

    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found: {XGB_MODEL_PATH}"
        )

    logistic_model = joblib.load(
        LOGISTIC_MODEL_PATH
    )

    xgb_model = joblib.load(
        XGB_MODEL_PATH
    )

    print("Logistic regression model loaded.")
    print("XGBoost model loaded.")

    return logistic_model, xgb_model


def get_model_estimator(
    model: object,
) -> object:
    """Extract the final estimator from a pipeline when required."""

    if hasattr(model, "steps"):
        steps = model.steps

        if not steps:
            raise ValueError(
                "Pipeline contains no steps."
            )

        return steps[-1][1]

    return model


def get_preprocessor(
    model: object,
) -> object | None:
    """Extract the preprocessing step from a pipeline."""

    if not hasattr(model, "named_steps"):
        return None

    named_steps = model.named_steps

    for _name, step in named_steps.items():
        if hasattr(
            step,
            "transformers_",
        ):
            return step

    return None


def get_transformed_feature_names(
    model: object,
    model_features: pd.DataFrame,
) -> list[str]:
    """
    Retrieve feature names after model preprocessing.

    Supports:
    - Pipeline + ColumnTransformer
    - ColumnTransformer
    - models trained directly on numeric data
    """

    preprocessor = get_preprocessor(model)

    if preprocessor is not None:
        try:
            names = (
                preprocessor
                .get_feature_names_out(
                    model_features.columns
                )
            )

            return [
                str(name)
                for name in names
            ]

        except (AttributeError, ValueError):
            pass

        try:
            names = (
                preprocessor
                .get_feature_names_out()
            )

            return [
                str(name)
                for name in names
            ]

        except (AttributeError, ValueError):
            pass

    estimator = get_model_estimator(model)

    if hasattr(
        estimator,
        "feature_names_in_",
    ):
        names = estimator.feature_names_in_

        return [
            str(name)
            for name in names
        ]

    if hasattr(
        model,
        "feature_names_in_",
    ):
        names = model.feature_names_in_

        return [
            str(name)
            for name in names
        ]

    return [
        str(column)
        for column in model_features.columns
    ]


def extract_logistic_importance(
    model: object,
    model_features: pd.DataFrame,
) -> pd.DataFrame:
    """Extract absolute and signed Logistic Regression coefficients."""

    estimator = get_model_estimator(model)

    if not hasattr(
        estimator,
        "coef_",
    ):
        raise ValueError(
            "Logistic Regression estimator does not contain coef_."
        )

    coefficients = np.asarray(
        estimator.coef_
    )

    if coefficients.ndim == 2:
        coefficients = coefficients[0]

    feature_names = get_transformed_feature_names(
        model,
        model_features,
    )

    if len(feature_names) != len(coefficients):
        raise ValueError(
            "Feature-name count does not match "
            f"coefficient count: "
            f"{len(feature_names)} != "
            f"{len(coefficients)}"
        )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(
                coefficients
            ),
        }
    )

    importance = importance.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    importance["rank"] = (
        importance.index + 1
    )

    return importance[
        [
            "rank",
            "feature",
            "coefficient",
            "absolute_coefficient",
        ]
    ]


def extract_xgb_importance(
    model: object,
    model_features: pd.DataFrame,
) -> pd.DataFrame:
    """Extract XGBoost feature importance."""

    estimator = get_model_estimator(model)

    if not hasattr(
        estimator,
        "feature_importances_",
    ):
        raise ValueError(
            "XGBoost model does not contain "
            "feature_importances_."
        )

    importances = np.asarray(
        estimator.feature_importances_
    )

    feature_names = get_transformed_feature_names(
        model,
        model_features,
    )

    if len(feature_names) != len(importances):
        raise ValueError(
            "Feature-name count does not match "
            f"importance count: "
            f"{len(feature_names)} != "
            f"{len(importances)}"
        )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    importance["rank"] = (
        importance.index + 1
    )

    return importance[
        [
            "rank",
            "feature",
            "importance",
        ]
    ]


def validate_importance(
    importance: pd.DataFrame,
    importance_column: str,
) -> None:
    """Validate generated feature-importance output."""

    if importance.empty:
        raise ValueError(
            "Feature importance dataframe is empty."
        )

    if importance["feature"].isna().any():
        raise ValueError(
            "Feature importance contains "
            "missing feature names."
        )

    if importance["feature"].duplicated().any():
        duplicate_count = int(
            importance["feature"]
            .duplicated()
            .sum()
        )

        raise ValueError(
            f"Feature importance contains "
            f"{duplicate_count} duplicate features."
        )

    if not np.isfinite(
        importance[importance_column]
    ).all():
        raise ValueError(
            "Feature importance contains "
            "non-finite values."
        )


def print_top_features(
    importance: pd.DataFrame,
    value_column: str,
    model_name: str,
    top_n: int = 20,
) -> None:
    """Print top model features."""

    print(
        f"\nTop {top_n} {model_name} features:"
    )

    print(
        importance[
            [
                "rank",
                "feature",
                value_column,
            ]
        ]
        .head(top_n)
        .to_string(index=False)
    )


def main() -> None:
    """Run model feature-importance analysis."""

    print("Loading feature dataset...")

    dataset = load_feature_dataset()

    model_features = prepare_model_features(
        dataset
    )

    print("\nLoading models...")

    logistic_model, xgb_model = load_models()

    print(
        "\nExtracting Logistic Regression importance..."
    )

    logistic_importance = (
        extract_logistic_importance(
            logistic_model,
            model_features,
        )
    )

    validate_importance(
        logistic_importance,
        "absolute_coefficient",
    )

    print_top_features(
        logistic_importance,
        "absolute_coefficient",
        "Logistic Regression",
    )

    print(
        "\nExtracting XGBoost importance..."
    )

    xgb_importance = extract_xgb_importance(
        xgb_model,
        model_features,
    )

    validate_importance(
        xgb_importance,
        "importance",
    )

    print_top_features(
        xgb_importance,
        "importance",
        "XGBoost",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logistic_importance.to_parquet(
        LOGISTIC_OUTPUT_PATH,
        index=False,
    )

    xgb_importance.to_parquet(
        XGB_OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSaved Logistic Regression importance to:"
    )
    print(LOGISTIC_OUTPUT_PATH)

    print(
        "\nSaved XGBoost importance to:"
    )
    print(XGB_OUTPUT_PATH)


if __name__ == "__main__":
    main()