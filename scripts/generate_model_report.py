from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROCESSED_DIR = Path("data/processed")

LOGISTIC_TEST_PATH = (
    PROCESSED_DIR / "baseline_test_predictions.parquet"
)

XGB_TEST_PATH = (
    PROCESSED_DIR / "xgb_test_predictions.parquet"
)

MODEL_COMPARISON_PATH = (
    PROCESSED_DIR / "model_comparison.parquet"
)

LOGISTIC_IMPORTANCE_PATH = (
    PROCESSED_DIR / "logistic_feature_importance.parquet"
)

XGB_IMPORTANCE_PATH = (
    PROCESSED_DIR / "xgb_feature_importance.parquet"
)

SHAP_IMPORTANCE_PATH = (
    PROCESSED_DIR / "xgb_shap_feature_importance.parquet"
)

JSON_OUTPUT_PATH = (
    PROCESSED_DIR / "model_report.json"
)

MARKDOWN_OUTPUT_PATH = (
    PROCESSED_DIR / "model_report.md"
)


LOGISTIC_THRESHOLD = 0.90
XGB_THRESHOLD = 0.45

TOP_N = 10


def load_predictions(
    path: Path,
    model_name: str,
) -> pd.DataFrame:
    """Load and validate model predictions."""
    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} predictions not found: {path}"
        )

    predictions = pd.read_parquet(path)

    probability_candidates = [
        "predicted_probability",
        "prediction_probability",
        "risk_probability",
        "probability",
    ]

    probability_column = next(
        (
            column
            for column in probability_candidates
            if column in predictions.columns
        ),
        None,
    )

    if probability_column is None:
        raise ValueError(
            f"{model_name}: probability column not found. "
            f"Available columns: {predictions.columns.tolist()}"
        )

    required_columns = {
        "subscriber_id",
        "risk_label",
        probability_column,
    }

    missing_columns = (
        required_columns - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{model_name}: missing columns: "
            f"{sorted(missing_columns)}"
        )

    predictions = predictions[
        [
            "subscriber_id",
            "risk_label",
            probability_column,
        ]
    ].copy()

    predictions = predictions.rename(
        columns={
            probability_column: "probability",
        }
    )

    predictions["probability"] = pd.to_numeric(
        predictions["probability"],
        errors="coerce",
    )

    predictions["risk_label"] = predictions[
        "risk_label"
    ].astype(int)

    if predictions["probability"].isna().any():
        raise ValueError(
            f"{model_name}: probability contains NaN values."
        )

    if not predictions["probability"].between(
        0,
        1,
    ).all():
        raise ValueError(
            f"{model_name}: probabilities must be "
            "between 0 and 1."
        )

    if predictions["subscriber_id"].duplicated().any():
        raise ValueError(
            f"{model_name}: duplicate subscriber IDs found."
        )

    return predictions


def calculate_metrics(
    predictions: pd.DataFrame,
    threshold: float,
) -> dict:
    """Calculate classification metrics at a threshold."""
    y_true = predictions["risk_label"].to_numpy()

    probabilities = predictions[
        "probability"
    ].to_numpy()

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "predicted_positive_rate": float(
            y_pred.mean()
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def load_top_features(
    path: Path,
    feature_column: str,
    importance_column: str,
    top_n: int = TOP_N,
) -> list[dict]:
    """Load top feature importance records."""
    if not path.exists():
        return []

    importance = pd.read_parquet(path)

    required_columns = {
        feature_column,
        importance_column,
    }

    missing_columns = (
        required_columns - set(importance.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{path}: missing columns: "
            f"{sorted(missing_columns)}"
        )

    importance = importance.sort_values(
        importance_column,
        ascending=False,
    ).head(top_n)

    return [
        {
            "feature": str(row[feature_column]),
            "importance": float(
                row[importance_column]
            ),
        }
        for _, row in importance.iterrows()
    ]


def build_report() -> dict:
    """Build the complete model evaluation report."""
    logistic = load_predictions(
        LOGISTIC_TEST_PATH,
        "Logistic Regression",
    )

    xgb = load_predictions(
        XGB_TEST_PATH,
        "XGBoost",
    )

    if set(logistic["subscriber_id"]) != set(
        xgb["subscriber_id"]
    ):
        raise ValueError(
            "Logistic Regression and XGBoost "
            "predictions contain different subscribers."
        )

    if not (
        logistic["risk_label"].reset_index(drop=True)
        == xgb["risk_label"].reset_index(drop=True)
    ).all():
        raise ValueError(
            "Target labels differ between model "
            "prediction datasets."
        )

    logistic_metrics = calculate_metrics(
        logistic,
        LOGISTIC_THRESHOLD,
    )

    xgb_metrics = calculate_metrics(
        xgb,
        XGB_THRESHOLD,
    )

    model_comparison = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                **logistic_metrics,
            },
            {
                "model": "xgboost",
                **xgb_metrics,
            },
        ]
    )

    best_model = model_comparison.loc[
        model_comparison["f1"].idxmax()
    ]

    logistic_importance = load_top_features(
        LOGISTIC_IMPORTANCE_PATH,
        feature_column="feature",
        importance_column="absolute_coefficient",
    )

    xgb_importance = load_top_features(
        XGB_IMPORTANCE_PATH,
        feature_column="feature",
        importance_column="importance",
    )

    shap_importance = load_top_features(
        SHAP_IMPORTANCE_PATH,
        feature_column="feature",
        importance_column="mean_abs_shap",
    )

    positive_rate = float(
        logistic["risk_label"].mean()
    )

    return {
        "dataset": {
            "test_rows": int(len(logistic)),
            "positive_rate": positive_rate,
            "positive_count": int(
                logistic["risk_label"].sum()
            ),
            "negative_count": int(
                (logistic["risk_label"] == 0).sum()
            ),
        },
        "models": {
            "logistic_regression": logistic_metrics,
            "xgboost": xgb_metrics,
        },
        "best_model_by_f1": {
            "model": str(best_model["model"]),
            "threshold": float(
                best_model["threshold"]
            ),
            "precision": float(
                best_model["precision"]
            ),
            "recall": float(
                best_model["recall"]
            ),
            "f1": float(best_model["f1"]),
            "pr_auc": float(
                best_model["pr_auc"]
            ),
            "roc_auc": float(
                best_model["roc_auc"]
            ),
        },
        "top_features": {
            "logistic_regression": logistic_importance,
            "xgboost": xgb_importance,
            "xgboost_shap": shap_importance,
        },
    }


def save_json(report: dict) -> None:
    """Save report as JSON."""
    with JSON_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print(
        f"Saved JSON report to:\n"
        f"{JSON_OUTPUT_PATH}"
    )


def format_metric(value: float) -> str:
    """Format a metric for Markdown."""
    return f"{value:.4f}"


def build_markdown(report: dict) -> str:
    """Build the Markdown model report."""
    dataset = report["dataset"]

    logistic = report["models"][
        "logistic_regression"
    ]

    xgb = report["models"]["xgboost"]

    best = report["best_model_by_f1"]

    lines = [
        "# AutoEvolve-X Model Evaluation Report",
        "",
        "## Dataset",
        "",
        f"- Test rows: {dataset['test_rows']:,}",
        (
            "- Positive samples: "
            f"{dataset['positive_count']:,}"
        ),
        (
            "- Negative samples: "
            f"{dataset['negative_count']:,}"
        ),
        (
            "- Positive rate: "
            f"{dataset['positive_rate']:.2%}"
        ),
        "",
        "## Model Comparison",
        "",
        (
            "| Model | Threshold | ROC-AUC | "
            "PR-AUC | Precision | Recall | F1 |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|"
        ),
        (
            "| Logistic Regression | "
            f"{logistic['threshold']:.2f} | "
            f"{format_metric(logistic['roc_auc'])} | "
            f"{format_metric(logistic['pr_auc'])} | "
            f"{format_metric(logistic['precision'])} | "
            f"{format_metric(logistic['recall'])} | "
            f"{format_metric(logistic['f1'])} |"
        ),
        (
            "| XGBoost | "
            f"{xgb['threshold']:.2f} | "
            f"{format_metric(xgb['roc_auc'])} | "
            f"{format_metric(xgb['pr_auc'])} | "
            f"{format_metric(xgb['precision'])} | "
            f"{format_metric(xgb['recall'])} | "
            f"{format_metric(xgb['f1'])} |"
        ),
        "",
        "## Best Model by F1",
        "",
        f"- Model: **{best['model']}**",
        f"- Threshold: **{best['threshold']:.2f}**",
        f"- Precision: **{best['precision']:.4f}**",
        f"- Recall: **{best['recall']:.4f}**",
        f"- F1: **{best['f1']:.4f}**",
        f"- PR-AUC: **{best['pr_auc']:.4f}**",
        f"- ROC-AUC: **{best['roc_auc']:.4f}**",
        "",
        "## Logistic Regression — Top Features",
        "",
        "| Rank | Feature | Importance |",
        "|---:|---|---:|",
    ]

    for rank, item in enumerate(
        report["top_features"][
            "logistic_regression"
        ],
        start=1,
    ):
        lines.append(
            f"| {rank} | "
            f"{item['feature']} | "
            f"{item['importance']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## XGBoost — Top Features",
            "",
            "| Rank | Feature | Importance |",
            "|---:|---|---:|",
        ]
    )

    for rank, item in enumerate(
        report["top_features"]["xgboost"],
        start=1,
    ):
        lines.append(
            f"| {rank} | "
            f"{item['feature']} | "
            f"{item['importance']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## XGBoost — Top SHAP Features",
            "",
            "| Rank | Feature | Mean Absolute SHAP |",
            "|---:|---|---:|",
        ]
    )

    for rank, item in enumerate(
        report["top_features"]["xgboost_shap"],
        start=1,
    ):
        lines.append(
            f"| {rank} | "
            f"{item['feature']} | "
            f"{item['importance']:.6f} |"
        )

    lines.append("")

    return "\n".join(lines)


def save_markdown(report: dict) -> None:
    """Save report as Markdown."""
    markdown = build_markdown(report)

    MARKDOWN_OUTPUT_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print(
        f"Saved Markdown report to:\n"
        f"{MARKDOWN_OUTPUT_PATH}"
    )


def main() -> None:
    """Generate the consolidated model report."""
    print("Generating model evaluation report...")

    report = build_report()

    print("\nModel summary:")

    for model_name, metrics in report[
        "models"
    ].items():
        print(
            f"  {model_name}: "
            f"F1={metrics['f1']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"PR-AUC={metrics['pr_auc']:.4f}"
        )

    print(
        "\nBest model by F1: "
        f"{report['best_model_by_f1']['model']}"
    )

    save_json(report)
    save_markdown(report)


if __name__ == "__main__":
    main()