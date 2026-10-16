from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "risk_label"
ID_COLUMN = "subscriber_id"


def build_baseline_pipeline(
    train_df: pd.DataFrame,
) -> Pipeline:
    """Build the preprocessing + logistic regression baseline."""

    feature_columns = [
        column
        for column in train_df.columns
        if column not in {TARGET_COLUMN, ID_COLUMN}
    ]

    categorical_columns = train_df[
        feature_columns
    ].select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_columns = [
        column
        for column in feature_columns
        if column not in categorical_columns
    ]

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="liblinear",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_baseline(
    train_df: pd.DataFrame,
) -> Pipeline:
    """Train the baseline model."""

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    if ID_COLUMN not in train_df.columns:
        raise ValueError(
            f"Missing identifier column: {ID_COLUMN}"
        )

    X_train = train_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_train = train_df[TARGET_COLUMN]

    pipeline = build_baseline_pipeline(train_df)

    pipeline.fit(X_train, y_train)

    return pipeline


def save_model(
    model: Pipeline,
    output_path: str | Path,
) -> None:
    """Save trained model to disk."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, output_path)