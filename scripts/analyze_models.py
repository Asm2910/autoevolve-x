from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROCESSED_DIR = Path("data/processed")

LOGISTIC_PREDICTIONS = (
    PROCESSED_DIR / "baseline_test_predictions.parquet"
)

XGB_PREDICTIONS = (
    PROCESSED_DIR / "xgb_test_predictions.parquet"
)

OUTPUT_PATH = PROCESSED_DIR / "model_comparison.parquet"


def load_predictions(
    path: Path,
    model_name: str,
) -> pd.DataFrame:
    """Load and normalize model prediction output."""

    if not path.exists():
        raise FileNotFoundError(
            f"{model_name}: prediction file not found: {path}"
        )

    df = pd.read_parquet(path)

    required_columns = {
        "subscriber_id",
        "risk_label",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{model_name}: missing required columns: "
            f"{sorted(missing)}"
        )

    probability_candidates = [
        "risk_probability",
        "predicted_probability",
        "prediction_probability",
        "probability",
        "prediction_prob",
    ]

    probability_column = next(
        (
            column
            for column in probability_candidates
            if column in df.columns
        ),
        None,
    )

    if probability_column is None:
        raise ValueError(
            f"{model_name}: no prediction probability column found. "
            f"Expected one of: {probability_candidates}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[
        [
            "subscriber_id",
            "risk_label",
            probability_column,
        ]
    ].copy()

    df = df.rename(
        columns={
            probability_column: "predicted_probability",
        }
    )

    if df["subscriber_id"].duplicated().any():
        raise ValueError(
            f"{model_name}: duplicate subscriber IDs found."
        )

    if df["risk_label"].isna().any():
        raise ValueError(
            f"{model_name}: missing target labels found."
        )

    if df["predicted_probability"].isna().any():
        raise ValueError(
            f"{model_name}: missing prediction probabilities found."
        )

    if not df["predicted_probability"].between(
        0.0,
        1.0,
    ).all():
        raise ValueError(
            f"{model_name}: prediction probabilities must "
            "be between 0 and 1."
        )

    if not df["risk_label"].isin([0, 1]).all():
        raise ValueError(
            f"{model_name}: risk_label must contain only 0 and 1."
        )

    print(
        f"{model_name}: loaded {len(df):,} predictions."
    )

    return df


def calculate_metrics(
    df: pd.DataFrame,
    threshold: float,
) -> dict[str, float]:
    """Calculate classification and ranking metrics."""

    y_true = df["risk_label"]
    probabilities = df["predicted_probability"]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "predicted_positive_rate": predictions.mean(),
    }


def validate_same_population(
    logistic: pd.DataFrame,
    xgb: pd.DataFrame,
) -> None:
    """Ensure both models are evaluated on the same subscribers."""

    logistic_ids = set(logistic["subscriber_id"])
    xgb_ids = set(xgb["subscriber_id"])

    if logistic_ids != xgb_ids:
        only_logistic = logistic_ids - xgb_ids
        only_xgb = xgb_ids - logistic_ids

        raise ValueError(
            "Models were evaluated on different subscribers. "
            f"Only in logistic: {len(only_logistic):,}; "
            f"only in XGBoost: {len(only_xgb):,}."
        )

    logistic = logistic.sort_values(
        "subscriber_id"
    ).reset_index(drop=True)

    xgb = xgb.sort_values(
        "subscriber_id"
    ).reset_index(drop=True)

    if not logistic["risk_label"].equals(
        xgb["risk_label"]
    ):
        raise ValueError(
            "Risk labels do not match between model outputs."
        )


def build_comparison(
    logistic: pd.DataFrame,
    xgb: pd.DataFrame,
) -> pd.DataFrame:
    """Build model-level comparison table."""

    logistic_threshold = 0.90
    xgb_threshold = 0.45

    logistic_metrics = calculate_metrics(
        logistic,
        logistic_threshold,
    )

    xgb_metrics = calculate_metrics(
        xgb,
        xgb_threshold,
    )

    rows = [
        {
            "model": "logistic_regression",
            "threshold": logistic_threshold,
            **logistic_metrics,
        },
        {
            "model": "xgboost",
            "threshold": xgb_threshold,
            **xgb_metrics,
        },
    ]

    return pd.DataFrame(rows)


def print_comparison(
    comparison: pd.DataFrame,
) -> None:
    """Print model comparison in a readable format."""

    print("\nModel comparison:")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nBest model by F1:")

    best_f1 = comparison.loc[
        comparison["f1"].idxmax()
    ]

    print(
        f"  model      = {best_f1['model']}"
    )
    print(
        f"  threshold  = {best_f1['threshold']:.2f}"
    )
    print(
        f"  precision  = {best_f1['precision']:.4f}"
    )
    print(
        f"  recall     = {best_f1['recall']:.4f}"
    )
    print(
        f"  f1         = {best_f1['f1']:.4f}"
    )
    print(
        f"  pr_auc     = {best_f1['pr_auc']:.4f}"
    )
    print(
        f"  roc_auc    = {best_f1['roc_auc']:.4f}"
    )


def main() -> None:
    """Run model comparison."""

    print("Loading test predictions...")

    logistic = load_predictions(
        LOGISTIC_PREDICTIONS,
        "Logistic regression",
    )

    xgb = load_predictions(
        XGB_PREDICTIONS,
        "XGBoost",
    )

    print("\nValidating model populations...")

    validate_same_population(
        logistic,
        xgb,
    )

    print("Population validation passed.")

    comparison = build_comparison(
        logistic,
        xgb,
    )

    print_comparison(comparison)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved model comparison to:\n"
        f"{OUTPUT_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()