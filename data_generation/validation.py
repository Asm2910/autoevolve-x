from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPECTED_FILES = {
    "customers": "customers.parquet",
    "usage_events": "usage_events.parquet",
    "recharges": "recharges.parquet",
    "network_events": "network_events.parquet",
    "risk_events": "risk_events.parquet",
    "labels": "labels.parquet",
}


REQUIRED_COLUMNS = {
    "customers": {
        "subscriber_id",
        "activation_date",
        "customer_segment",
        "plan_type",
        "region",
        "device_type",
        "tenure_days",
        "age_band",
        "preferred_language",
        "account_status",
    },
    "usage_events": {
        "event_id",
        "subscriber_id",
        "timestamp",
        "call_count",
        "call_duration_sec",
        "sms_count",
        "data_mb",
        "international_calls",
        "short_calls",
        "unique_destinations",
        "night_activity",
        "roaming_flag",
    },
    "recharges": {
        "recharge_id",
        "subscriber_id",
        "timestamp",
        "recharge_amount",
        "payment_channel",
        "recharge_type",
        "successful",
    },
    "network_events": {
        "network_event_id",
        "subscriber_id",
        "timestamp",
        "cell_id",
        "network_type",
        "signal_strength",
        "latency_ms",
        "packet_loss_pct",
        "data_session_minutes",
        "roaming_flag",
        "failed_connection",
    },
    "risk_events": {
        "risk_event_id",
        "subscriber_id",
        "timestamp",
        "risk_type",
        "risk_score",
        "risk_level",
        "trigger_count",
        "source_system",
    },
    "labels": {
        "subscriber_id",
        "risk_label",
    },
}


EXPECTED_ID_COLUMNS = {
    "customers": "subscriber_id",
    "usage_events": "event_id",
    "recharges": "recharge_id",
    "network_events": "network_event_id",
    "risk_events": "risk_event_id",
    "labels": "subscriber_id",
}


def _record_error(errors: list[str], message: str) -> None:
    errors.append(message)


def _check_required_columns(
    name: str,
    dataframe: pd.DataFrame,
    errors: list[str],
) -> None:
    missing = REQUIRED_COLUMNS[name] - set(dataframe.columns)

    if missing:
        _record_error(
            errors,
            f"{name}: missing columns: {sorted(missing)}",
        )


def _check_nulls(
    name: str,
    dataframe: pd.DataFrame,
    errors: list[str],
) -> None:
    null_counts = dataframe.isna().sum()
    null_columns = null_counts[null_counts > 0]

    if not null_columns.empty:
        details = {
            column: int(count)
            for column, count in null_columns.items()
        }

        _record_error(
            errors,
            f"{name}: null values found: {details}",
        )


def _check_duplicate_ids(
    name: str,
    dataframe: pd.DataFrame,
    errors: list[str],
) -> None:
    id_column = EXPECTED_ID_COLUMNS[name]

    if id_column not in dataframe.columns:
        return

    duplicate_count = int(
        dataframe[id_column].duplicated().sum()
    )

    if duplicate_count > 0:
        _record_error(
            errors,
            (
                f"{name}: {duplicate_count} duplicate "
                f"{id_column} values found"
            ),
        )


def _check_customers(
    customers: pd.DataFrame,
    errors: list[str],
) -> None:
    valid_segments = {
        "consumer",
        "business",
        "premium",
    }

    valid_plans = {
        "prepaid",
        "postpaid",
    }

    valid_account_statuses = {
        "active",
        "suspended",
        "inactive",
    }

    if not customers["customer_segment"].isin(
        valid_segments
    ).all():
        _record_error(
            errors,
            "customers: invalid customer_segment values found",
        )

    if not customers["plan_type"].isin(valid_plans).all():
        _record_error(
            errors,
            "customers: invalid plan_type values found",
        )

    if not customers["account_status"].isin(
        valid_account_statuses
    ).all():
        _record_error(
            errors,
            "customers: invalid account_status values found",
        )

    if (customers["tenure_days"] < 0).any():
        _record_error(
            errors,
            "customers: negative tenure_days found",
        )

    activation_dates = pd.to_datetime(
        customers["activation_date"],
        errors="coerce",
    )

    if activation_dates.isna().any():
        _record_error(
            errors,
            "customers: invalid activation_date values found",
        )


def _check_usage_events(
    usage_events: pd.DataFrame,
    errors: list[str],
) -> None:
    non_negative_columns = [
        "call_count",
        "call_duration_sec",
        "sms_count",
        "data_mb",
        "international_calls",
        "short_calls",
        "unique_destinations",
    ]

    for column in non_negative_columns:
        if (usage_events[column] < 0).any():
            _record_error(
                errors,
                f"usage_events: negative values in {column}",
            )

    for column in [
        "night_activity",
        "roaming_flag",
    ]:
        if not usage_events[column].isin({0, 1}).all():
            _record_error(
                errors,
                f"usage_events: invalid binary values in {column}",
            )

    invalid_international = (
        usage_events["international_calls"]
        > usage_events["call_count"]
    ).any()

    if invalid_international:
        _record_error(
            errors,
            (
                "usage_events: international_calls exceeds "
                "call_count"
            ),
        )

    invalid_short = (
        usage_events["short_calls"]
        > usage_events["call_count"]
    ).any()

    if invalid_short:
        _record_error(
            errors,
            "usage_events: short_calls exceeds call_count",
        )


def _check_recharges(
    recharges: pd.DataFrame,
    errors: list[str],
) -> None:
    if (recharges["recharge_amount"] <= 0).any():
        _record_error(
            errors,
            "recharges: non-positive recharge_amount found",
        )

    if not recharges["successful"].isin({0, 1}).all():
        _record_error(
            errors,
            "recharges: invalid successful values found",
        )

    valid_channels = {
        "mobile_app",
        "ussd",
        "retail_store",
        "bank",
        "web",
    }

    if not recharges["payment_channel"].isin(
        valid_channels
    ).all():
        _record_error(
            errors,
            "recharges: invalid payment_channel values found",
        )


def _check_network_events(
    network_events: pd.DataFrame,
    errors: list[str],
) -> None:
    if not network_events["network_type"].isin(
        {"3G", "4G", "5G"}
    ).all():
        _record_error(
            errors,
            "network_events: invalid network_type values found",
        )

    if (
        network_events["latency_ms"] < 0
    ).any():
        _record_error(
            errors,
            "network_events: negative latency_ms found",
        )

    if (
        network_events["packet_loss_pct"] < 0
    ).any():
        _record_error(
            errors,
            "network_events: negative packet_loss_pct found",
        )

    if (
        network_events["data_session_minutes"] < 0
    ).any():
        _record_error(
            errors,
            (
                "network_events: negative "
                "data_session_minutes found"
            ),
        )

    for column in [
        "roaming_flag",
        "failed_connection",
    ]:
        if not network_events[column].isin({0, 1}).all():
            _record_error(
                errors,
                (
                    f"network_events: invalid binary "
                    f"values in {column}"
                ),
            )


def _check_risk_events(
    risk_events: pd.DataFrame,
    errors: list[str],
) -> None:
    if (
        risk_events["risk_score"] < 0
    ).any() or (
        risk_events["risk_score"] > 100
    ).any():
        _record_error(
            errors,
            "risk_events: risk_score outside [0, 100]",
        )

    if (
        risk_events["trigger_count"] < 0
    ).any():
        _record_error(
            errors,
            "risk_events: negative trigger_count found",
        )

    valid_levels = {
        "low",
        "medium",
        "high",
        "critical",
    }

    if not risk_events["risk_level"].isin(
        valid_levels
    ).all():
        _record_error(
            errors,
            "risk_events: invalid risk_level values found",
        )


def _check_labels(
    labels: pd.DataFrame,
    customers: pd.DataFrame,
    errors: list[str],
) -> None:
    if not labels["risk_label"].isin({0, 1}).all():
        _record_error(
            errors,
            "labels: risk_label must contain only 0 and 1",
        )

    customer_ids = set(
        customers["subscriber_id"]
    )

    label_ids = set(
        labels["subscriber_id"]
    )

    missing_labels = customer_ids - label_ids
    extra_labels = label_ids - customer_ids

    if missing_labels:
        _record_error(
            errors,
            (
                "labels: subscribers missing labels: "
                f"{len(missing_labels)}"
            ),
        )

    if extra_labels:
        _record_error(
            errors,
            (
                "labels: unknown subscribers found: "
                f"{len(extra_labels)}"
            ),
        )

    expected_positive_rate = labels[
        "risk_label"
    ].mean()

    if not 0 < expected_positive_rate < 1:
        _record_error(
            errors,
            "labels: risk_label contains only one class",
        )


def _check_referential_integrity(
    customers: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    errors: list[str],
) -> None:
    customer_ids = set(
        customers["subscriber_id"]
    )

    for name in [
        "usage_events",
        "recharges",
        "network_events",
        "risk_events",
        "labels",
    ]:
        dataframe = datasets[name]

        subscriber_ids = set(
            dataframe["subscriber_id"]
        )

        unknown_ids = subscriber_ids - customer_ids

        if unknown_ids:
            _record_error(
                errors,
                (
                    f"{name}: {len(unknown_ids)} "
                    "unknown subscriber_id values"
                ),
            )


def _check_timestamps(
    name: str,
    dataframe: pd.DataFrame,
    errors: list[str],
    start_date: str,
    end_date: str,
) -> None:
    if "timestamp" not in dataframe.columns:
        return

    timestamps = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
    )

    if timestamps.isna().any():
        _record_error(
            errors,
            f"{name}: invalid timestamp values found",
        )
        return

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    outside_range = (
        (timestamps < start)
        | (timestamps >= end)
    ).sum()

    if outside_range > 0:
        _record_error(
            errors,
            (
                f"{name}: {outside_range} timestamps "
                "outside configured time range"
            ),
        )


def validate_datasets(
    data_dir: str | Path,
    start_date: str = "2026-01-01",
    end_date: str = "2026-06-30",
) -> None:
    """Validate all generated datasets."""

    data_dir = Path(data_dir)
    errors: list[str] = []

    datasets: dict[str, pd.DataFrame] = {}

    # --------------------------------------------------------------
    # File existence and loading
    # --------------------------------------------------------------

    for name, filename in EXPECTED_FILES.items():
        path = data_dir / filename

        if not path.exists():
            _record_error(
                errors,
                f"{name}: file not found: {path}",
            )
            continue

        try:
            datasets[name] = pd.read_parquet(path)
        except Exception as exc:
            _record_error(
                errors,
                f"{name}: failed to read parquet: {exc}",
            )

    if errors:
        raise ValueError(
            "Dataset validation failed:\n- "
            + "\n- ".join(errors)
        )

    # --------------------------------------------------------------
    # Individual dataset checks
    # --------------------------------------------------------------

    for name, dataframe in datasets.items():
        _check_required_columns(
            name,
            dataframe,
            errors,
        )

        _check_nulls(
            name,
            dataframe,
            errors,
        )

        _check_duplicate_ids(
            name,
            dataframe,
            errors,
        )

    customers = datasets["customers"]
    usage_events = datasets["usage_events"]
    recharges = datasets["recharges"]
    network_events = datasets["network_events"]
    risk_events = datasets["risk_events"]
    labels = datasets["labels"]

    _check_customers(
        customers,
        errors,
    )

    _check_usage_events(
        usage_events,
        errors,
    )

    _check_recharges(
        recharges,
        errors,
    )

    _check_network_events(
        network_events,
        errors,
    )

    _check_risk_events(
        risk_events,
        errors,
    )

    _check_labels(
        labels,
        customers,
        errors,
    )

    # --------------------------------------------------------------
    # Referential integrity
    # --------------------------------------------------------------

    _check_referential_integrity(
        customers,
        datasets,
        errors,
    )

    # --------------------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------------------

    for name in [
        "usage_events",
        "recharges",
        "network_events",
        "risk_events",
    ]:
        _check_timestamps(
            name,
            datasets[name],
            errors,
            start_date,
            end_date,
        )

    # --------------------------------------------------------------
    # Final result
    # --------------------------------------------------------------

    if errors:
        raise ValueError(
            "Dataset validation failed:\n- "
            + "\n- ".join(errors)
        )

    print("Dataset validation passed.")
    print()
    print("Dataset summary:")

    for name, dataframe in datasets.items():
        print(
            f"  {name:<15} "
            f"rows={len(dataframe):>8,} "
            f"columns={len(dataframe.columns):>3}"
        )

    print()
    print(
        "Label positive rate: "
        f"{labels['risk_label'].mean():.4f}"
    )