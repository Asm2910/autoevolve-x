# Data dictionary

The model reads 254 features for each subscriber on each snapshot date. This page explains where they come from, how they are named and what each group means. The exact column list is in `data/processed/feature_manifest.json`.

## Rows and label

One row is one active subscriber on one snapshot date.

| Column | Meaning |
| --- | --- |
| `subscriber_id` | Identifier of the subscriber |
| `snapshot_date` | The date the subscriber is scored. Nine month-end snapshots, from 2025-09-30 to 2026-05-31 |
| `risk_label` | 1 if fraud is **confirmed** within 30 days after the snapshot date, otherwise 0 |

Subscribers whose fraud was already confirmed before the snapshot date are left out, because their status is already known. About 0.9% of rows have `risk_label = 1`. The label is the *confirmed* outcome, so fraud that is never detected counts as 0, and a small share of innocent subscribers are wrongly confirmed (label noise by design).

The identifier and label columns are never used as model inputs.

## How features are built

Every feature uses only events that happened **before or on** the snapshot date, so nothing from the future leaks in.

| Pattern | Meaning |
| --- | --- |
| `<group>__<metric>__7d`, `__30d`, `__90d` | The metric over the last 7, 30 or 90 days |
| `<group>__<metric>_trend__7d_vs_90d`, `__30d_vs_90d` | A shorter window compared with the 90-day window, to show a recent change |
| `<group>__has_events__<window>` | 1 if the subscriber had any events of that kind in the window, otherwise 0 |

The `has_events` flags matter: when a subscriber had no events, counts are set to 0 but averages stay missing, and the flag tells the model which case it is looking at.

## Feature groups

| Group | Features | What it describes |
| --- | --- | --- |
| `customer` | 8 | Who the subscriber is: plan, region, device, tenure |
| `usage` | 63 | Calls, texts and data use |
| `recharge` | 68 | Top-ups and payments |
| `network` | 59 | Signal quality, sessions and the cells the device connects to |
| `risk` | 56 | Events raised by the operator's own risk detection |

Windowed metrics are written once below. Each one exists for the 7, 30 and 90-day windows unless noted.

### customer (8, no windows)

Seven are categorical and one is numeric.

| Column | Type | Meaning |
| --- | --- | --- |
| `customer__customer_segment` | category | Commercial segment of the subscriber |
| `customer__plan_type` | category | Plan the subscriber is on |
| `customer__region` | category | Home region |
| `customer__device_type` | category | Type of device |
| `customer__tenure_days` | number | Days from activation to the snapshot date |
| `customer__age_band` | category | Age band of the account holder |
| `customer__preferred_language` | category | Preferred language |
| `customer__account_status` | category | Status of the account |

### usage

| Metric | Meaning |
| --- | --- |
| `total_usage_events` | Number of usage records |
| `total_calls`, `total_sms`, `total_data_mb` | Calls made, texts sent, megabytes used |
| `total_call_duration_sec` | Total time on calls, in seconds |
| `total_international_calls`, `total_short_calls` | International calls, and calls recorded as short |
| `avg_call_duration_sec`, `avg_calls_per_event`, `avg_sms_per_event`, `avg_data_mb_per_event` | Averages per usage record |
| `avg_unique_destinations`, `max_unique_destinations` | Average and highest number of distinct people called per record |
| `night_activity_rate` | Share of records that happen at night |
| `roaming_rate` | Share of records made while roaming |
| `international_call_ratio`, `short_call_ratio` | International calls and short calls as a share of all calls |
| `avg_call_duration_per_call` | Total call time divided by number of calls |
| Trends | `total_calls`, `total_international_calls` and `total_short_calls`, as 7d vs 90d and 30d vs 90d |

### recharge

| Metric | Meaning |
| --- | --- |
| `total_recharges`, `successful_recharges`, `failed_recharges` | Number of top-up attempts, and how many succeeded or failed |
| `recharge_success_rate` | Successful share of attempts |
| `total_recharge_amount`, `avg_recharge_amount`, `median_recharge_amount`, `max_recharge_amount`, `min_recharge_amount`, `recharge_amount_std` | Size of top-ups: total, average, median, largest, smallest, spread |
| `avg_amount_per_successful_recharge` | Total amount divided by successful top-ups |
| `payment_channel_rate_<channel>` | Share of top-ups made through `bank`, `mobile_app`, `retail_store`, `ussd` or `web` |
| `recharge_type_rate_<type>` | Share of top-ups of type `advance_payment`, `bill_payment`, `bundle`, `emergency` or `standard` |
| Trends | `total_recharges`, as 7d vs 90d and 30d vs 90d |

### network

| Metric | Meaning |
| --- | --- |
| `total_network_events` | Number of network records |
| `avg_signal_strength`, `min_signal_strength`, `max_signal_strength` | Signal strength: average, weakest, strongest |
| `avg_latency_ms`, `max_latency_ms` | Delay on the connection, in milliseconds |
| `avg_packet_loss_pct`, `max_packet_loss_pct` | Share of data packets lost |
| `total_session_minutes`, `avg_session_minutes` | Time spent in data sessions |
| `roaming_rate` | Share of records made while roaming |
| `connection_failure_rate`, `poor_signal_rate` | Share of records with a failed connection, or with poor signal |
| `network_type_rate_3G`, `_4G`, `_5G` | Share of records on each network generation |
| `unique_cells`, `top_cell_share` | Number of distinct cell towers used, and the share of records on the most-used one. A device that never moves sits on very few cells |
| Trends | `total_network_events`, as 7d vs 90d and 30d vs 90d |

### risk

These come from the operator's own anomaly detection, so they are signals about the subscriber, not the answer.

| Metric | Meaning |
| --- | --- |
| `total_risk_events` | Number of risk events raised |
| `avg_risk_score`, `max_risk_score`, `min_risk_score`, `risk_score_std` | Average, highest, lowest and spread of the risk score |
| `avg_trigger_count`, `max_trigger_count` | How many rules fired per event |
| `high_risk_event_count`, `critical_risk_event_count` | Events rated high, or critical |
| `high_risk_event_rate`, `critical_risk_event_rate` | Those events as a share of all risk events |
| `risk_type_rate_<type>` | Share of events of type `account_anomaly`, `behavioral_anomaly`, `location_anomaly`, `network_anomaly`, `payment_anomaly` or `usage_anomaly` |
| Trends | `total_risk_events`, as 7d vs 90d and 30d vs 90d |

## The world behind the data

The data is synthetic, so these settings are known (from `configs/data.yaml`).

| Setting | Value |
| --- | --- |
| Customers | 25,000 |
| Time range | 2025-07-01 to 2026-06-30 |
| Events | 1.2 million usage, 300,000 recharge, 600,000 network, 30,000 risk |
| Share of subscribers who commit fraud | 10% |
| Share of fraud that is ever confirmed | 85% |
| Median days from fraud start to confirmation | 14 |
| Innocent subscribers wrongly confirmed | 0.4% |
| Fraud types | SIM box, subscription fraud, account takeover, wangiri (the mix shifts over time) |
| Label horizon | 30 days |

Fraud is decided first and events are generated from it, so a model cannot read the label off the features. Only about 0.9% of snapshot rows are positive, because confirmation is delayed, only part of fraud is confirmed, and known fraud is excluded.
