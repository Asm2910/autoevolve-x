"""Estimators that scikit-learn does not provide in the shape the arena needs."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest


class IsolationForestScorer(ClassifierMixin, BaseEstimator):
    """An unsupervised anomaly detector that speaks the classifier interface.

    Labels are never used: the forest learns what "ordinary" looks like and scores how
    isolated a row is. The raw anomaly score is turned into the share of training rows that
    are *less* anomalous (an empirical CDF), so the output is a rank score in [0, 1]. That is
    all PR-AUC and the top-k review metrics need. It is not a probability.

    It is the arena's reality check: if a supervised model cannot beat an unsupervised one,
    the labels are adding little.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: int = 1024,
        max_features: float = 1.0,
        random_state: int = 0,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):  # noqa: N803 - scikit-learn naming
        self.forest_ = IsolationForest(
            n_estimators=int(self.n_estimators),
            max_samples=int(min(self.max_samples, len(X))),
            max_features=float(self.max_features),
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(X)
        self.reference_ = np.sort(-self.forest_.score_samples(X))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):  # noqa: N803
        anomaly = -self.forest_.score_samples(X)
        rank = np.searchsorted(self.reference_, anomaly, side="right") / len(self.reference_)
        rank = np.clip(rank, 0.0, 1.0)
        return np.column_stack([1.0 - rank, rank])

    def predict(self, X):  # noqa: N803
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
