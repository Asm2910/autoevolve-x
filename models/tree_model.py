from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

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


def prepare_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare model features and target."""

    required_columns = {
        TARGET_COLUMN,
        ID_COLUMN,
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    data = dataframe.copy()

    target = data[TARGET_COLUMN].astype(int)

    features = data.drop(
        columns=[
            TARGET_COLUMN,
            ID_COLUMN,
        ]
    )

    categorical_columns = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in features.columns
    ]

    features = pd.get_dummies(
        features,
        columns=categorical_columns,
        drop_first=False,
        dtype=float,
    )

    non_numeric_columns = features.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    if non_numeric_columns:
        raise ValueError(
            "Non-numeric features found: "
            f"{non_numeric_columns}"
        )

    features = features.astype(float)

    return features, target


def train_xgb_model(
    train: pd.DataFrame,
) -> XGBClassifier:
    """Train XGBoost classifier."""

    features, target = prepare_features(train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        features,
        target,
    )

    return model


def align_features_with_model(
    model: XGBClassifier,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Align features to the exact training feature schema."""

    model_features = model.get_booster().feature_names

    if not model_features:
        raise ValueError(
            "Model does not contain feature names."
        )

    model_features = list(model_features)

    # Add columns that are missing in this dataset.
    missing_features = [
        column
        for column in model_features
        if column not in features.columns
    ]

    for column in missing_features:
        features[column] = 0.0

    # Remove columns that were not present during training.
    extra_features = [
        column
        for column in features.columns
        if column not in model_features
    ]

    if extra_features:
        features = features.drop(
            columns=extra_features
        )

    # Exact same order as training.
    features = features.loc[
        :,
        model_features,
    ]

    return features.astype(float)


def predict_proba(
    model: XGBClassifier,
    features: pd.DataFrame,
) -> pd.Series:
    """Generate positive-class probabilities."""

    aligned_features = align_features_with_model(
        model,
        features,
    )

    probabilities = model.predict_proba(
        aligned_features
    )[:, 1]

    return pd.Series(
        probabilities,
        index=features.index,
        name="predicted_probability",
    )


def save_model(
    model: XGBClassifier,
    path: str | Path,
) -> None:
    """Save trained model."""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        path,
    )


def load_model(
    path: str | Path,
) -> XGBClassifier:
    """Load trained model."""

    return joblib.load(path)