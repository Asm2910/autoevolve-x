from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

DATA_PATH = Path("data/processed/features.parquet")
MODEL_PATH = Path("models/artifacts/xgboost_classifier.joblib")
OUTPUT_DIR = Path("data/processed")

SAMPLE_SIZE = 3_000
RANDOM_STATE = 42
TOP_N = 20


TARGET_COLUMN = "risk_label"
ID_COLUMN = "subscriber_id"


def load_feature_dataset() -> pd.DataFrame:
    """Load and validate the complete feature dataset."""
    print("Loading feature dataset...")

    df = pd.read_parquet(DATA_PATH)

    required_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    print(f"Feature dataset loaded: {df.shape}")

    return df


def load_model():
    """Load the trained XGBoost model."""
    print("\nLoading XGBoost model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print("XGBoost model loaded.")

    return model


def get_model_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create the original model input feature dataset."""
    excluded_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    features = df[feature_columns].copy()

    return features


def transform_features(
    features: pd.DataFrame,
    model,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Transform raw model features into the exact feature space
    expected by the trained XGBoost model.
    """

    categorical_columns = features.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numerical_columns = [
        column
        for column in features.columns
        if column not in categorical_columns
    ]

    numeric_features = features[numerical_columns].copy()

    categorical_features = features[categorical_columns].copy()

    categorical_encoded = pd.get_dummies(
        categorical_features,
        columns=categorical_columns,
        dtype=float,
    )

    transformed = pd.concat(
        [
            numeric_features,
            categorical_encoded,
        ],
        axis=1,
    )

    transformed = transformed.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    transformed.columns = transformed.columns.astype(str)

    # The trained model stores the expected feature names
    # in the underlying booster.
    booster = model.get_booster()

    model_feature_names = booster.feature_names

    if model_feature_names is None:
        raise ValueError(
            "XGBoost model does not contain feature names."
        )

    model_feature_names = [
        str(name)
        for name in model_feature_names
    ]

    # Add missing model features.
    missing_features = [
        feature
        for feature in model_feature_names
        if feature not in transformed.columns
    ]

    for feature in missing_features:
        transformed[feature] = 0.0

    # Remove unexpected features.
    transformed = transformed.reindex(
        columns=model_feature_names,
        fill_value=0.0,
    )

    transformed = transformed.astype(float)

    if list(transformed.columns) != model_feature_names:
        raise ValueError(
            "Transformed features do not match the "
            "feature order expected by the XGBoost model."
        )

    print(
        f"Transformed feature shape: {transformed.shape}"
    )

    return transformed, model_feature_names


def create_shap_explainer(model):
    """Create a SHAP TreeExplainer for XGBoost."""
    return shap.TreeExplainer(model)


def calculate_shap_values(
    explainer,
    features: pd.DataFrame,
):
    """Calculate SHAP values."""
    print("\nCalculating SHAP values...")

    shap_values = explainer.shap_values(features)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    if shap_values.shape != features.shape:
        raise ValueError(
            "SHAP values shape does not match "
            f"feature shape. SHAP={shap_values.shape}, "
            f"features={features.shape}"
        )

    print(
        f"SHAP values shape: {shap_values.shape}"
    )

    return shap_values


def build_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Build global SHAP feature importance."""
    importance = np.abs(shap_values).mean(axis=0)

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": importance,
        }
    )

    result = result.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    result.insert(
        0,
        "rank",
        np.arange(1, len(result) + 1),
    )

    return result


def save_feature_importance(
    importance: pd.DataFrame,
) -> Path:
    """Save global SHAP feature importance."""
    output_path = (
        OUTPUT_DIR / "xgb_shap_feature_importance.parquet"
    )

    importance.to_parquet(
        output_path,
        index=False,
    )

    print(
        "\nSaved SHAP feature importance to:"
    )
    print(output_path)

    return output_path


def save_shap_values(
    shap_values: np.ndarray,
    feature_names: list[str],
    sample_ids: pd.Series,
) -> Path:
    """Save row-level SHAP values."""
    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_names,
    )

    shap_df.insert(
        0,
        ID_COLUMN,
        sample_ids.reset_index(drop=True),
    )

    output_path = (
        OUTPUT_DIR / "xgb_shap_values.parquet"
    )

    shap_df.to_parquet(
        output_path,
        index=False,
    )

    print(
        "Saved row-level SHAP values to:"
    )
    print(output_path)

    return output_path


def print_top_features(
    importance: pd.DataFrame,
) -> None:
    """Print the most important SHAP features."""
    print(
        f"\nTop {TOP_N} XGBoost SHAP features:"
    )

    print(
        importance.head(TOP_N).to_string(
            index=False
        )
    )


def main() -> None:
    """Run XGBoost SHAP analysis."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_feature_dataset()
    model = load_model()

    features = get_model_features(df)

    print(
        f"\nOriginal model features: "
        f"{features.shape[1]}"
    )

    sample_size = min(
        SAMPLE_SIZE,
        len(features),
    )

    sample_indices = (
        features.sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        ).index
    )

    sample_features = features.loc[
        sample_indices
    ].copy()

    sample_ids = df.loc[
        sample_indices,
        ID_COLUMN,
    ]

    print(
        f"\nUsing SHAP sample: "
        f"{len(sample_features):,} rows"
    )

    transformed_features, feature_names = (
        transform_features(
            sample_features,
            model,
        )
    )

    print(
        f"Model feature shape: "
        f"{transformed_features.shape}"
    )

    explainer = create_shap_explainer(model)

    shap_values = calculate_shap_values(
        explainer,
        transformed_features,
    )

    importance = build_feature_importance(
        shap_values,
        feature_names,
    )

    print_top_features(importance)

    save_feature_importance(importance)

    save_shap_values(
        shap_values,
        feature_names,
        sample_ids,
    )


if __name__ == "__main__":
    main()