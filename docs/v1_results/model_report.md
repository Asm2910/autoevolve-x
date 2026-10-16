# AutoEvolve-X Model Evaluation Report

## Dataset

- Test rows: 3,000
- Positive samples: 150
- Negative samples: 2,850
- Positive rate: 5.00%

## Model Comparison

| Model | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.90 | 0.9616 | 0.5554 | 0.4919 | 0.6067 | 0.5433 |
| XGBoost | 0.45 | 0.9674 | 0.5726 | 0.6019 | 0.4133 | 0.4901 |

## Best Model by F1

- Model: **logistic_regression**
- Threshold: **0.90**
- Precision: **0.4919**
- Recall: **0.6067**
- F1: **0.5433**
- PR-AUC: **0.5554**
- ROC-AUC: **0.9616**

## Logistic Regression — Top Features

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | numerical__total_calls | 1.344923 |
| 2 | numerical__total_usage_events | 1.341702 |
| 3 | numerical__total_data_mb | 1.309600 |
| 4 | numerical__short_call_ratio | 1.175006 |
| 5 | numerical__max_risk_score | 1.162909 |
| 6 | numerical__recharge_type_rate_bill_payment | 1.156941 |
| 7 | numerical__min_risk_score | 1.089109 |
| 8 | numerical__avg_data_mb_per_event | 1.045093 |
| 9 | categorical__account_status_active | 0.966860 |
| 10 | numerical__total_short_calls | 0.925394 |

## XGBoost — Top Features

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | avg_risk_score | 0.112182 |
| 2 | total_short_calls | 0.092393 |
| 3 | max_risk_score | 0.061678 |
| 4 | failed_recharges | 0.025269 |
| 5 | total_international_calls | 0.024572 |
| 6 | connection_failure_rate | 0.018970 |
| 7 | avg_latency_ms | 0.016828 |
| 8 | avg_signal_strength | 0.016488 |
| 9 | avg_unique_destinations | 0.014580 |
| 10 | min_risk_score | 0.014361 |

## XGBoost — Top SHAP Features

| Rank | Feature | Mean Absolute SHAP |
|---:|---|---:|
| 1 | avg_risk_score | 0.972635 |
| 2 | max_risk_score | 0.710802 |
| 3 | total_short_calls | 0.674902 |
| 4 | min_risk_score | 0.442362 |
| 5 | total_international_calls | 0.369257 |
| 6 | connection_failure_rate | 0.318888 |
| 7 | short_call_ratio | 0.308359 |
| 8 | night_activity_rate | 0.307635 |
| 9 | avg_latency_ms | 0.232050 |
| 10 | avg_signal_strength | 0.188589 |
