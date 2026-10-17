"""Per-prediction explanations from XGBoost's built-in exact tree contributions.

`pred_contribs=True` returns SHAP values for tree models (log-odds units), so no
extra dependency is needed. Contributions of one-hot columns are summed back onto
the original feature, and for every row:

    bias + sum(contributions) == log-odds of the predicted probability
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def supports_explanations(pipeline) -> bool:
    return isinstance(pipeline, Pipeline) and isinstance(pipeline[-1], XGBClassifier)


def _source_features(pipeline: Pipeline) -> list[str]:
    """Original feature name for every column the XGBoost model sees."""
    preprocessor = pipeline.named_steps["preprocess"]
    sources: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        columns = list(columns)
        if name == "categorical":
            for column, categories in zip(columns, transformer.categories_, strict=True):
                sources.extend([column] * len(categories))
        else:
            sources.extend(columns)
    return sources


def contributions(pipeline: Pipeline, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Log-odds contribution of each original feature, plus the per-row bias."""
    sources = _source_features(pipeline)
    matrix = np.asarray(pipeline.named_steps["preprocess"].transform(frame), dtype=np.float32)
    booster = pipeline[-1].get_booster()
    raw = booster.predict(xgb.DMatrix(matrix), pred_contribs=True)
    if raw.shape[1] != len(sources) + 1:
        raise ValueError("contribution columns do not match the preprocessed features")

    per_column = pd.DataFrame(raw[:, :-1], columns=sources)
    per_feature = per_column.T.groupby(level=0, sort=False).sum().T
    return per_feature, raw[:, -1]


def top_reasons(
    pipeline: Pipeline, frame: pd.DataFrame, top_k: int = 5, min_contribution: float = 1e-3
) -> list[list[dict]]:
    """For each row, the features pushing the risk score up the most."""
    per_feature, _ = contributions(pipeline, frame)
    result = []
    for i in range(len(frame)):
        row = per_feature.iloc[i]
        best = row[row > min_contribution].sort_values(ascending=False).head(top_k)
        reasons = []
        for feature, value in best.items():
            raw_value = frame.iloc[i][feature]
            raw_value = raw_value.item() if hasattr(raw_value, "item") else raw_value
            reasons.append(
                {
                    "feature": feature,
                    "value": None if pd.isna(raw_value) else raw_value,
                    "contribution": round(float(value), 4),
                }
            )
        result.append(reasons)
    return result
