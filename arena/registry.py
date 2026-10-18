"""The catalogue of algorithms the platform can train.

Everything that needs to know "which models exist" reads this file: the pipeline builder,
the experiment spec (allowed names and hyperparameter bounds), the arena and the agent
prompt. Adding an algorithm means adding one entry here and one block in configs/models.yaml.

Each family declares how its inputs must be prepared:

  scaled   median-impute and standardise numbers, sparse one-hot (linear models)
  imputed  median-impute numbers, dense one-hot (scikit-learn forests, which gain nothing
           from scaling but cannot take missing values)
  native   leave numbers alone, dense one-hot (boosters that handle missing values)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# name -> (minimum, maximum, integer?)
Bounds = dict[str, tuple[float, float, bool]]


@dataclass(frozen=True)
class ModelFamily:
    name: str
    kind: str  # linear | bagging | boosting | anomaly
    preprocessing: str  # scaled | imputed | native
    make: Callable[[dict, int], object]
    bounds: Bounds
    summary: str


# ---- factories (imports are lazy so unused libraries cost nothing) -------------------------
def _logistic_regression(params: dict, seed: int):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(random_state=seed, **params)


def _random_forest(params: dict, seed: int):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_jobs=-1, random_state=seed, class_weight="balanced_subsample", **params)


def _extra_trees(params: dict, seed: int):
    from sklearn.ensemble import ExtraTreesClassifier

    # "balanced_subsample" needs bootstrap sampling, which extra-trees does not use by default
    return ExtraTreesClassifier(n_jobs=-1, random_state=seed, class_weight="balanced", **params)


def _xgboost(params: dict, seed: int):
    from xgboost import XGBClassifier

    return XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=seed, **params)


def _lightgbm(params: dict, seed: int):
    from lightgbm import LGBMClassifier

    # subsample only takes effect when subsample_freq > 0
    return LGBMClassifier(subsample_freq=1, n_jobs=-1, random_state=seed, verbose=-1, **params)


def _hist_gradient_boosting(params: dict, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **params)


def _catboost(params: dict, seed: int):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        random_seed=seed, thread_count=-1, verbose=0, allow_writing_files=False, **params
    )


def _isolation_forest(params: dict, seed: int):
    from arena.estimators import IsolationForestScorer

    return IsolationForestScorer(random_state=seed, **params)


REGISTRY: dict[str, ModelFamily] = {
    family.name: family
    for family in [
        ModelFamily(
            "logistic_regression", "linear", "scaled", _logistic_regression,
            {"C": (0.001, 10.0, False)},
            "Regularised linear baseline. Fast, transparent, cannot model interactions.",
        ),
        ModelFamily(
            "random_forest", "bagging", "imputed", _random_forest,
            {
                "n_estimators": (50, 600, True),
                "max_depth": (3, 20, True),
                "min_samples_leaf": (1, 100, True),
                "max_features": (0.02, 1.0, False),
            },
            "Averaged deep trees. Robust and low-variance, but large and slow on wide data.",
        ),
        ModelFamily(
            "extra_trees", "bagging", "imputed", _extra_trees,
            {
                "n_estimators": (50, 600, True),
                "max_depth": (3, 20, True),
                "min_samples_leaf": (1, 100, True),
                "max_features": (0.02, 1.0, False),
            },
            "Random-split forest. Even lower variance than a random forest, often a little weaker.",
        ),
        ModelFamily(
            "xgboost", "boosting", "native", _xgboost,
            {
                "n_estimators": (50, 600, True),
                "max_depth": (2, 8, True),
                "learning_rate": (0.01, 0.3, False),
                "subsample": (0.5, 1.0, False),
                "colsample_bytree": (0.2, 1.0, False),
                "min_child_weight": (1, 20, True),
            },
            "Gradient-boosted trees with exact contribution-based explanations.",
        ),
        ModelFamily(
            "lightgbm", "boosting", "native", _lightgbm,
            {
                "n_estimators": (50, 600, True),
                "num_leaves": (4, 64, True),
                "learning_rate": (0.01, 0.3, False),
                "subsample": (0.5, 1.0, False),
                "colsample_bytree": (0.2, 1.0, False),
                "min_child_samples": (5, 200, True),
            },
            "Leaf-wise gradient boosting. Usually the fastest booster to train.",
        ),
        ModelFamily(
            "hist_gradient_boosting", "boosting", "native", _hist_gradient_boosting,
            {
                "max_iter": (50, 600, True),
                "max_depth": (2, 8, True),
                "learning_rate": (0.01, 0.3, False),
                "min_samples_leaf": (5, 200, True),
                "l2_regularization": (0.0, 10.0, False),
            },
            "scikit-learn's histogram booster. No extra dependency, no built-in explanations.",
        ),
        ModelFamily(
            "catboost", "boosting", "native", _catboost,
            {
                "iterations": (50, 600, True),
                "depth": (2, 8, True),
                "learning_rate": (0.01, 0.3, False),
                "l2_leaf_reg": (1.0, 20.0, False),
            },
            "Ordered boosting with symmetric trees. Strong on small signals, slower to train.",
        ),
        ModelFamily(
            "isolation_forest", "anomaly", "imputed", _isolation_forest,
            {
                "n_estimators": (50, 500, True),
                "max_samples": (64, 4096, True),
                "max_features": (0.1, 1.0, False),
            },
            "Unsupervised anomaly detector that never sees labels. A reality check, not a contender.",
        ),
    ]
}


def names() -> list[str]:
    return list(REGISTRY)


def get_family(name: str) -> ModelFamily:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model: {name!r}. Available: {', '.join(REGISTRY)}") from None


def param_bounds() -> dict[str, Bounds]:
    return {name: family.bounds for name, family in REGISTRY.items()}


def _preprocessor(mode: str, numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    if mode == "scaled":
        numeric_step = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        categorical_step = OneHotEncoder(handle_unknown="ignore")
    elif mode == "imputed":
        numeric_step = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True))])
        categorical_step = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif mode == "native":
        numeric_step = "passthrough"
        categorical_step = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")
    return ColumnTransformer([("numeric", numeric_step, numeric), ("categorical", categorical_step, categorical)])


def build_pipeline(name: str, params: dict, numeric: list[str], categorical: list[str], seed: int) -> Pipeline:
    """Preprocessing and model in one artifact, so serving needs no extra code."""
    family = get_family(name)
    return Pipeline(
        [
            ("preprocess", _preprocessor(family.preprocessing, numeric, categorical)),
            ("model", family.make(dict(params), seed)),
        ]
    )
