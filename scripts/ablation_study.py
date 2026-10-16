"""Feature-group ablation using cross-validation on train + validation only."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.evaluate_models import build_xgb_pipeline

PROCESSED = Path("data/processed")
ID_COL, LABEL = "subscriber_id", "risk_label"
SEED = 42

GROUP_STARTS = [
    ("customer", "customer_segment"),
    ("usage", "total_usage_events"),
    ("recharge", "total_recharges"),
    ("network", "total_network_events"),
    ("risk", "total_risk_events"),
]


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    starts = [columns.index(first) for _, first in GROUP_STARTS] + [len(columns)]
    assert starts == sorted(starts), "feature groups are not in column order"
    groups = {
        name: columns[starts[i] : starts[i + 1]]
        for i, (name, _) in enumerate(GROUP_STARTS)
    }
    assert sorted(c for cols in groups.values() for c in cols) == sorted(columns)
    return groups


def build_logistic(x: pd.DataFrame) -> Pipeline:
    categorical = x.select_dtypes(include=["object", "category", "string"]).columns
    numeric = [c for c in x.columns if c not in categorical]
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                list(categorical),
            ),
        ]
    )
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def main() -> None:
    train = pd.read_parquet(PROCESSED / "train.parquet")
    validation = pd.read_parquet(PROCESSED / "validation.parquet")
    df = pd.concat([train, validation], ignore_index=True)
    y = df[LABEL]
    x = df.drop(columns=[ID_COL, LABEL])

    groups = feature_groups(list(x.columns))
    configs = {"all": list(x.columns)}
    for name, cols in groups.items():
        configs[f"drop_{name}"] = [c for c in x.columns if c not in cols]
        configs[f"only_{name}"] = cols

    builders = {
        "logistic_regression": lambda xs, ys: build_logistic(xs),
        "xgboost": build_xgb_pipeline,
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows = []
    for model_name, builder in builders.items():
        for config, cols in configs.items():
            xs = x[cols]
            result = cross_validate(
                builder(xs, y),
                xs,
                y,
                cv=cv,
                scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
                n_jobs=1,
            )
            rows.append(
                {
                    "model": model_name,
                    "config": config,
                    "n_features": len(cols),
                    "pr_auc": result["test_pr_auc"].mean(),
                    "pr_auc_std": result["test_pr_auc"].std(),
                    "roc_auc": result["test_roc_auc"].mean(),
                }
            )
            print(f"{model_name:20s} {config:16s} PR-AUC {rows[-1]['pr_auc']:.3f}")

    table = pd.DataFrame(rows)
    baseline = table[table["config"] == "all"].set_index("model")["pr_auc"]
    table["delta_pr_auc_vs_all"] = table["pr_auc"] - table["model"].map(baseline)
    table.to_csv(PROCESSED / "ablation_results.csv", index=False)

    positive_rate = y.mean()
    print(f"\nChance-level PR-AUC (positive rate): {positive_rate:.3f}")
    for model_name in builders:
        print(f"\n=== {model_name} ===")
        subset = table[table["model"] == model_name].drop(columns="model")
        print(subset.round(3).to_string(index=False))


if __name__ == "__main__":
    main()