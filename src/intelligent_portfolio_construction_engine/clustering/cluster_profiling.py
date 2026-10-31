import pandas as pd

from intelligent_portfolio_construction_engine.models.cluster_profile import (
    ClusterProfile,
)

FEATURE_COLUMNS = [
    "volatility",
    "current_drawdown",
    "cagr",
    "momentum_factor",
    "beta",
    "sharpe_ratio",
    "sortino_ratio",
]


def build_cluster_profiles(features, labels) -> list[ClusterProfile]:
    clustered_features = features.copy()
    clustered_features["cluster_id"] = labels.to_numpy()

    profiles = []

    for cluster_id, cluster_data in clustered_features.groupby("cluster_id"):
        feature_data = cluster_data[FEATURE_COLUMNS]

        profile = ClusterProfile(
            cluster_id=int(cluster_id),
            asset_count=len(cluster_data),
            centroid={
                column: float(feature_data[column].mean())
                for column in FEATURE_COLUMNS
            },
            median_profile={
                column: float(feature_data[column].median())
                for column in FEATURE_COLUMNS
            },
        )

        profiles.append(profile)

    return profiles