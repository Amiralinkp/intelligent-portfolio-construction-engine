
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "volatility",
    "current_drawdown",
    "cagr",
    "momentum_factor",
    "beta",
    "sharpe_ratio",
    "sortino_ratio",
]

GROUP_COLUMNS = [
    "snapshot_date",
    "model_version",
    "cluster_id",
]


def analyze_cluster_characteristics(behavioral_history):
    required_columns = {
        *GROUP_COLUMNS,
        "symbol",
        *FEATURE_COLUMNS}

    missing_columns = required_columns - set(behavioral_history.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}")

    if behavioral_history.empty:
        raise ValueError("Behavioral history cannot be empty.")

    data = behavioral_history.copy()

    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"])

    if data["snapshot_date"].isna().any():
        raise ValueError("Snapshot dates cannot be missing.")

    if data[GROUP_COLUMNS].isna().any().any():
        raise ValueError("Grouping columns cannot contain missing values.")

    if data["symbol"].isna().any():
        raise ValueError("Asset symbols cannot be missing.")

    if data.duplicated(["snapshot_date", "symbol"]).any():
        raise ValueError("Duplicate asset snapshots detected.")

    feature_values = data[FEATURE_COLUMNS].to_numpy(
        dtype=float
    )

    if not np.isfinite(feature_values).all():
        raise ValueError("Behavioral features must be finite.")

    grouped = data.groupby(
        GROUP_COLUMNS,
        sort=True,
        observed=True,
    )

    characteristics = grouped[FEATURE_COLUMNS].agg(
        ["mean", "median"]
    )

    characteristics.columns = [
        f"{feature}_{statistic}"
        for feature, statistic in characteristics.columns
    ]

    characteristics["asset_count"] = grouped["symbol"].nunique()

    return characteristics.reset_index()
