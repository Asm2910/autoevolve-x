"""Fair, reproducible comparison of the logistic and XGBoost models."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

PROCESSED = Path("data/processed")
ARTIFACTS = Path("models/artifacts")
ID_COL, LABEL = "subscriber_id", "risk_label"
SEED = 42
N_BOOT = 1000
GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED / f"{name}.parquet")
    return df.drop(columns=[ID_COL, LABEL]), df[LABEL]


def build_xgb_pipeline(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    categorical = x_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric = [c for c in x_train.columns if c not in categorical]
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            ("num", "passthrough", numeric),
        ]
    )
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", classifier)])


def sweep_thresholds(y: np.ndarray, p: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for t in GRID:
        pred = (p >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
            }
        )
    table = pd.DataFrame(rows)
    return float(table.loc[table["f1"].idxmax(), "threshold"]), table


def point_metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    pred = (p >= t).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def bootstrap(y, probs: dict[str, np.ndarray], thresholds: dict[str, float]):
    rng = np.random.default_rng(SEED)
    n = len(y)
    keys = ["pr_auc", "precision", "recall", "f1"]
    draws = {m: {k: [] for k in keys} for m in probs}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0:
            continue
        for m, p in probs.items():
            pb = p[idx]
            pred = (pb >= thresholds[m]).astype(int)
            draws[m]["pr_auc"].append(average_precision_score(yb, pb))
            draws[m]["precision"].append(precision_score(yb, pred, zero_division=0))
            draws[m]["recall"].append(recall_score(yb, pred, zero_division=0))
            draws[m]["f1"].append(f1_score(yb, pred, zero_division=0))
    return {m: {k: np.array(v) for k, v in d.items()} for m, d in draws.items()}


def ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> None:
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("validation")
    x_test, y_test = load_split("test")
    y_val, y_test = y_val.to_numpy(), y_test.to_numpy()

    logistic = joblib.load(ARTIFACTS / "baseline_logistic_regression.joblib")
    xgb = build_xgb_pipeline(x_train, y_train)
    xgb.fit(x_train, y_train)
    joblib.dump(xgb, ARTIFACTS / "xgboost_pipeline.joblib")

    models = {"logistic_regression": logistic, "xgboost": xgb}
    val_probs = {m: mdl.predict_proba(x_val)[:, 1] for m, mdl in models.items()}
    test_probs = {m: mdl.predict_proba(x_test)[:, 1] for m, mdl in models.items()}

    thresholds, sweeps = {}, {}
    for m in models:
        thresholds[m], sweeps[m] = sweep_thresholds(y_val, val_probs[m])
        edge = thresholds[m] in (GRID[0], GRID[-1])
        print(f"{m}: best validation threshold = {thresholds[m]:.2f}"
              f"{'  <-- EDGE OF GRID' if edge else ''}")
        sweeps[m].to_parquet(PROCESSED / f"{m}_threshold_sweep_v2.parquet", index=False)

    draws = bootstrap(y_test, test_probs, thresholds)
    results = {}
    for m in models:
        point = point_metrics(y_test, test_probs[m], thresholds[m])
        results[m] = {
            "threshold": thresholds[m],
            "validation_f1_at_threshold": float(
                sweeps[m].loc[sweeps[m]["threshold"] == thresholds[m], "f1"].iloc[0]
            ),
            "test": point,
            "test_ci95": {k: ci(draws[m][k]) for k in draws[m]},
        }

    a, b = "logistic_regression", "xgboost"
    results["paired_difference_xgb_minus_logistic"] = {
        k: {
            "mean": float((draws[b][k] - draws[a][k]).mean()),
            "ci95": ci(draws[b][k] - draws[a][k]),
        }
        for k in ["pr_auc", "f1"]
    }
    results["test_positives"] = int(y_test.sum())

    out = PROCESSED / "model_evaluation_v2.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"\nTest positives: {int(y_test.sum())}")
    for m in models:
        r = results[m]
        t = r["test"]
        print(f"\n{m}  (threshold {r['threshold']:.2f})")
        for k in ["roc_auc", "pr_auc", "precision", "recall", "f1"]:
            interval = r["test_ci95"].get(k)
            extra = f"  95% CI [{interval[0]:.3f}, {interval[1]:.3f}]" if interval else ""
            print(f"  {k:10s} {t[k]:.3f}{extra}")
    print("\nPaired difference (XGBoost - Logistic):")
    for k, v in results["paired_difference_xgb_minus_logistic"].items():
        print(f"  {k:7s} mean {v['mean']:+.3f}  95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    print("  (if the CI contains 0, the models are statistically tied)")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()