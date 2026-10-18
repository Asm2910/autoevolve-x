"""Deterministic experiment execution. No language model is involved here.

Given an ExperimentSpec this trains the model on the TRAIN snapshots, scores the
VALIDATION snapshots, and compares against the current champion with a paired
bootstrap (the same resampled rows for both). The TEST snapshots are never read
here: the agent must not tune on data it will be judged on.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import average_precision_score

from agentkit.data_access import load_split
from agents.spec import ExperimentSpec
from models.training import (
    LABEL,
    PROCESSED_DIR,
    build_pipeline,
    interval,
    select_features,
)


class ExperimentRunner:
    def __init__(self, config: dict, bootstrap_samples: int | None = None) -> None:
        self.config = config
        self.seed = config["training"]["seed"]
        self.n_boot = bootstrap_samples or config["agent"]["bootstrap_samples"]
        self.manifest = json.loads((PROCESSED_DIR / "feature_manifest.json").read_text())
        self.train = load_split("train", PROCESSED_DIR)
        self.validation = load_split("validation", PROCESSED_DIR)
        self.y_val = self.validation[LABEL].to_numpy()
        self.champion: ExperimentSpec | None = None
        self.champion_probs: np.ndarray | None = None

    def baseline_spec(self) -> ExperimentSpec:
        """The configured model on every feature group and window."""
        return ExperimentSpec(
            model="xgboost",
            groups=list(self.config["features"]["groups"]),
            windows=list(self.config["features"]["windows"]),
            hypothesis="Baseline: all feature groups and windows",
        )

    def _fit_predict(self, spec: ExperimentSpec) -> tuple[np.ndarray, int]:
        features = select_features(self.manifest, spec.groups, spec.windows)
        categorical = [c for c in features if c in self.manifest["categorical"]]
        numeric = [c for c in features if c not in categorical]
        params = spec.resolved_params(self.config["models"][spec.model])
        pipeline = build_pipeline(spec.model, params, numeric, categorical, self.seed)
        pipeline.fit(self.train[features], self.train[LABEL])
        return pipeline.predict_proba(self.validation[features])[:, 1], len(features)

    def set_champion(self, spec: ExperimentSpec) -> None:
        self.champion = spec
        self.champion_probs, _ = self._fit_predict(spec)

    def run(self, spec: ExperimentSpec) -> dict:
        """Train the candidate and compare it with the champion on validation data."""
        if self.champion is None:
            raise RuntimeError("set_champion() must be called before run()")
        probs, n_features = self._fit_predict(spec)

        rng = np.random.default_rng(self.seed)  # same rows for every comparison
        n = len(self.y_val)
        candidate_draws = np.empty(self.n_boot)
        champion_draws = np.empty(self.n_boot)
        for i in range(self.n_boot):
            idx = rng.integers(0, n, n)
            candidate_draws[i] = average_precision_score(self.y_val[idx], probs[idx])
            champion_draws[i] = average_precision_score(self.y_val[idx], self.champion_probs[idx])
        diff = candidate_draws - champion_draws

        return {
            "n_features": n_features,
            "val_pr_auc": float(average_precision_score(self.y_val, probs)),
            "champion_val_pr_auc": float(average_precision_score(self.y_val, self.champion_probs)),
            "diff_mean": float(diff.mean()),
            "diff_ci": interval(diff),
        }