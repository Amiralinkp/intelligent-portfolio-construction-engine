
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "snapshot_date",
    "symbol",
    "cluster_id",
]


def analyze_market_behavior_distribution(
    behavioral_history: pd.DataFrame,
) -> pd.DataFrame:
    missing_columns = set(REQUIRED_COLUMNS) - set(
        behavioral_history.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if behavioral_history.empty:
        raise ValueError("Behavioral history cannot be empty.")

    data = behavioral_history[REQUIRED_COLUMNS].copy()

    data["snapshot_date"] = pd.to_datetime(
        data["snapshot_date"]
    )

    if data.isna().any().any():
        raise ValueError(
            "Required behavioral columns cannot contain missing values."
        )

    if data.duplicated(["snapshot_date", "symbol"]).any():
        raise ValueError("Duplicate asset snapshots detected.")

    distribution = (
        data.groupby(
            ["snapshot_date", "cluster_id"],
            sort=True,
            observed=True,
        )
        .agg(
            asset_count=("symbol", "nunique"),
        )
        .reset_index()
    )

    distribution["total_assets"] = (
        distribution.groupby("snapshot_date")[
            "asset_count"
        ].transform("sum")
    )

    distribution["cluster_share"] = (
        distribution["asset_count"]
        / distribution["total_assets"]
    )

    distribution["cluster_count"] = (
        distribution.groupby("snapshot_date")[
            "cluster_id"
        ].transform("nunique")
    )

    distribution["normalized_entropy"] = np.nan

    for snapshot_date, indices in distribution.groupby(
        "snapshot_date"
    ).groups.items():
        shares = distribution.loc[
            indices, "cluster_share"
        ].to_numpy()

        cluster_count = len(shares)

        if cluster_count > 1:
            entropy = -np.sum(
                shares * np.log(shares)
            )

            normalized_entropy = (
                entropy / np.log(cluster_count)
            )

            distribution.loc[
                indices, "normalized_entropy"
            ] = normalized_entropy

        else:
            distribution.loc[
                indices, "normalized_entropy"
            ] = 0.0

    distribution["dominant_cluster_share"] = (
        distribution.groupby("snapshot_date")[
            "cluster_share"
        ].transform("max")
    )

    return distribution.reset_index(drop=True)
