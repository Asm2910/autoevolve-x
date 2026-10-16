from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path(
    "models/artifacts/baseline_logistic_regression.joblib"
)

DEFAULT_THRESHOLD = 0.90


class RiskModel:
    """Production inference wrapper for the trained risk model."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
    ) -> None:
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """Load the trained model artifact."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {self.model_path}"
            )

        return joblib.load(self.model_path)

    def _prepare_features(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Validate and prepare inference features."""

        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame.")

        if features.empty:
            raise ValueError("features cannot be empty.")

        model_features = features.copy()

        if "subscriber_id" in model_features.columns:
            model_features = model_features.drop(
                columns=["subscriber_id"]
            )

        if "risk_label" in model_features.columns:
            model_features = model_features.drop(
                columns=["risk_label"]
            )

        return model_features

    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """Return positive-class probabilities."""

        model_features = self._prepare_features(features)

        probabilities = self.model.predict_proba(
            model_features
        )[:, 1]

        return np.asarray(probabilities)

    def predict(
        self,
        features: pd.DataFrame,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> pd.DataFrame:
        """Generate risk predictions."""

        if not 0 < threshold < 1:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        probabilities = self.predict_proba(features)

        predictions = (
            probabilities >= threshold
        ).astype(int)

        result = pd.DataFrame(
            {
                "risk_probability": probabilities,
                "risk_prediction": predictions,
            },
            index=features.index,
        )

        if "subscriber_id" in features.columns:
            result.insert(
                0,
                "subscriber_id",
                features["subscriber_id"].values,
            )

        return result