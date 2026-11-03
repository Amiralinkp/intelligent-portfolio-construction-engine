from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from intelligent_portfolio_construction_engine.clustering.model_selection import (
    ClusteringModelSelector,
)
from intelligent_portfolio_construction_engine.models.clustering_model_artifact import (
    ClusteringModelArtifact,
)


class WalkForwardModelTrainer:

    def __init__(
        self,
        feature_columns: tuple[str, ...],
        min_k: int,
        max_k: int,
    ) -> None:
        if not feature_columns:
            raise ValueError("Feature columns cannot be empty.")

        if min_k < 2 or max_k < min_k:
            raise ValueError("Invalid cluster range.")

        self.feature_columns = feature_columns
        self.min_k = min_k
        self.max_k = max_k

    def fit(
        self,
        snapshots: pd.DataFrame,
        training_cutoff: pd.Timestamp,
        model_version: str,
    ) -> tuple[ClusteringModelArtifact, dict[str, Any]]:

        required_columns = [
            "snapshot_date",
            *self.feature_columns,
        ]

        missing = set(required_columns).difference(snapshots.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        history = snapshots.copy()
        history["snapshot_date"] = pd.to_datetime(
            history["snapshot_date"]
        )

        cutoff = pd.Timestamp(training_cutoff)

        training_data = history.loc[
            history["snapshot_date"] <= cutoff
        ].copy()

        if training_data.empty:
            raise ValueError("No training observations available.")

        training_data = training_data.sort_values(
            "snapshot_date",
            ignore_index=True,
        )

        features = training_data.loc[
            :, list(self.feature_columns)
        ]

        if features.isna().any().any():
            raise ValueError("Training features contain missing values.")

        values = features.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                "Training features must contain finite numeric values."
            )

        if len(features) <= self.max_k:
            raise ValueError(
                "Insufficient observations for the configured cluster range."
            )

        scaler = RobustScaler()
        scaled_values = scaler.fit_transform(features)

        scaled_features = pd.DataFrame(
            scaled_values,
            columns=self.feature_columns,
            index=training_data.index,
        )

        selector = ClusteringModelSelector(
            min_k=self.min_k,
            max_k=self.max_k,
        )

        best_result, candidates = selector.select(scaled_features)

        estimator = best_result.clustering_result.estimator

        if estimator is None:
            raise RuntimeError(
                "Selected clustering model has no fitted estimator."
            )

        artifact = ClusteringModelArtifact(
            estimator=estimator,
            scaler=scaler,
            feature_columns=self.feature_columns,
            model_name=best_result.model_name,
            n_clusters=best_result.n_clusters,
            training_cutoff=cutoff,
            model_version=model_version,
            metadata={
                "snapshot_count": len(training_data),
                "candidate_count": len(candidates),
                "composite_score": float(
                    best_result.composite_score
                ),
                "metrics": {
                    key: float(value)
                    for key, value in best_result.metrics.items()
                },
            },
        )

        diagnostics = {
            "training_snapshot_count": len(training_data),
            "model_name": best_result.model_name,
            "n_clusters": best_result.n_clusters,
            "candidate_count": len(candidates),
            "composite_score": best_result.composite_score,
            "metrics": best_result.metrics,
        }

        return artifact, diagnostics