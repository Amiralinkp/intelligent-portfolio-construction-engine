import pandas as pd
import numpy as np
from intelligent_portfolio_construction_engine.features.behavioral_profiling import CLUSTERING_FEATURES
import json
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from intelligent_portfolio_construction_engine.clustering.model_selection import ClusteringModelSelector


SNAPSHOT_METADATA_COLUMNS = ["symbol", "asset_class"]

SNAPSHOT_COLUMNS = [*SNAPSHOT_METADATA_COLUMNS, *CLUSTERING_FEATURES]


def combine_behavioral_snapshots(snapshots: list[pd.DataFrame]):
    if not snapshots:
        return pd.DataFrame(columns=["snapshot_date", *SNAPSHOT_COLUMNS])

    normalized_snapshots = []

    for snapshot in snapshots:
        if snapshot.empty:
            continue

        normalized = snapshot.copy()
        normalized.index = pd.to_datetime(normalized.index)
        normalized.index.name = "snapshot_date"

        normalized_snapshots.append(normalized)

    if not normalized_snapshots:
        return pd.DataFrame(columns=["snapshot_date", *SNAPSHOT_COLUMNS])

    combined = pd.concat(normalized_snapshots)
    combined = combined.reset_index()

    combined = combined.sort_values(by=["snapshot_date", "symbol"], ignore_index=True)

    return combined[["snapshot_date", *SNAPSHOT_COLUMNS]]




def cluster_and_save_behavioral_history(snapshots: pd.DataFrame, settings, output_dir: Path):
    if snapshots.empty:
        raise ValueError("No behavioral snapshots available for clustering.")

    required_columns = ["snapshot_date", "symbol", "asset_class", *CLUSTERING_FEATURES]
    missing_columns = set(required_columns).difference(snapshots.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required snapshot columns: {sorted(missing_columns)}"
        )

    history = snapshots.copy()
    history["snapshot_date"] = pd.to_datetime(history["snapshot_date"])
    history = history.sort_values(
        ["snapshot_date", "symbol"],
        ignore_index=True,
    )

    features = history[CLUSTERING_FEATURES]

    if not features.notna().all().all():
        raise ValueError("Behavioral features contain missing values.")

    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Behavioral features must be finite numeric values.")

    if len(history) <= settings.CLUSTERING_MAX_K:
        raise ValueError("Not enough snapshots for the configured cluster range.")

    snapshot_ids = (
        history["symbol"].astype(str)
        + "|"
        + history["snapshot_date"].astype(str))

    if snapshot_ids.duplicated().any():
        raise ValueError("Duplicate asset/date snapshots found.")

    scaled_values = RobustScaler().fit_transform(features)
    scaled_features = pd.DataFrame(scaled_values, columns=CLUSTERING_FEATURES, index=snapshot_ids)

    selector = ClusteringModelSelector(
        min_k=settings.CLUSTERING_MIN_K,
        max_k=settings.CLUSTERING_MAX_K,)
    
    best_result, _ = selector.select(scaled_features)

    history["cluster_id"] = best_result.clustering_result.labels.reindex(snapshot_ids).to_numpy()

    output_dir.mkdir(parents=True, exist_ok=True)

    history.to_csv(output_dir / "historical_behavioral_states.csv", index=False)

    metadata = {
        "model_name": best_result.model_name,
        "n_clusters": best_result.n_clusters,
        "metrics": best_result.metrics,
        "composite_score": best_result.composite_score,
        "features": CLUSTERING_FEATURES,
        "snapshot_count": len(history),}

    with (output_dir / "behavioral_clustering_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return history
