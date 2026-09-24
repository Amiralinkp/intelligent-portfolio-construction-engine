import pandas as pd
from intelligent_portfolio_construction_engine.models.cluster_profile import ClusterProfile


FEATURE_COLUMNS = [
    "volatility",
    "current_drawdown",
    "cagr",
    "momentum_factor",
    "beta",
    "sharpe_ratio",
    "sortino_ratio"]


def build_cluster_profiles(features, labels):

    clustered_features = features.copy()
    clustered_features["cluster_id"] = labels.to_numpy()

    profiles = []

    for cluster_id, cluster_data in clustered_features.groupby("cluster_id"):

        profile = ClusterProfile(
            cluster_id=int(cluster_id),
            asset_count=len(cluster_data),
            **{column: float(cluster_data[column].median())
                for column in FEATURE_COLUMNS},)

        profiles.append(profile)

    return profiles