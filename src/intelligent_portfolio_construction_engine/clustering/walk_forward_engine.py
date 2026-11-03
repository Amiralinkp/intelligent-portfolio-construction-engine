from dataclasses import replace
from typing import Any
import numpy as np
import pandas as pd
from intelligent_portfolio_construction_engine.clustering.cluster_alignment import (
    ClusterAlignmentEngine,
)
from intelligent_portfolio_construction_engine.clustering.cluster_profiling import (
    FEATURE_COLUMNS,
    build_cluster_profiles,
)
from intelligent_portfolio_construction_engine.clustering.model_version_registry import (
    ModelVersionRegistry,
)
from intelligent_portfolio_construction_engine.clustering.walk_forward_schedule import (
    WalkForwardSchedule,
)
from intelligent_portfolio_construction_engine.clustering.walk_forward_trainer import (
    WalkForwardModelTrainer,
)
from intelligent_portfolio_construction_engine.models.asset_cluster_assignment import (
    AssetClusterAssignment,
)
from intelligent_portfolio_construction_engine.models.cluster_alignment_result import (
    ClusterAlignmentResult,
)
from intelligent_portfolio_construction_engine.models.clustering_model_artifact import (
    ClusteringModelArtifact,
)
from intelligent_portfolio_construction_engine.models.walk_forward_result import (
    WalkForwardResult,
)


class WalkForwardEngine:
    def __init__(
        self,
        registry: ModelVersionRegistry,
        threshold_estimator: Any,
        min_k: int = 2,
        max_k: int = 5,
        retraining_interval_weeks: int = 52,
        minimum_training_weeks: int = 104,
    ) -> None:
        if threshold_estimator is None:
            raise ValueError("A threshold estimator is required.")

        self.feature_columns = tuple(FEATURE_COLUMNS)
        self.registry = registry
        self.threshold_estimator = threshold_estimator

        self.schedule = WalkForwardSchedule(
            retraining_interval_weeks=retraining_interval_weeks,
            minimum_training_weeks=minimum_training_weeks,
        )

        self.trainer = WalkForwardModelTrainer(
            feature_columns=self.feature_columns,
            min_k=min_k,
            max_k=max_k,
        )

        self.alignment_engine = ClusterAlignmentEngine()

    def run(self, snapshots: pd.DataFrame) -> WalkForwardResult:
        history = self._validate_snapshots(snapshots)

        origins = self.schedule.generate(
            history["snapshot_date"]
        )

        if not origins:
            raise ValueError(
                "No training origins available for the supplied history."
            )

        assigned_history: list[pd.DataFrame] = []
        diagnostics: list[dict[str, Any]] = []
        alignment_results: dict[str, ClusterAlignmentResult] = {}

        previous_artifact: ClusteringModelArtifact | None = None

        for index, origin in enumerate(origins):
            model_version = f"wf-v{origin.version}"

            artifact, model_diagnostics = self.trainer.fit(
                snapshots=history,
                training_cutoff=origin.training_cutoff,
                model_version=model_version,
            )

            if previous_artifact is None:
                cluster_mapping = {
                    cluster_id: cluster_id
                    for cluster_id in range(artifact.n_clusters)
                }
            else:
                alignment = self._align_models(
                    previous_artifact=previous_artifact,
                    current_artifact=artifact,
                    history=history,
                    cutoff=origin.training_cutoff,
                )

                alignment_results[model_version] = alignment

                previous_mapping = self._read_mapping(
                    previous_artifact
                )

                cluster_mapping = self._build_cluster_mapping(
                    alignment=alignment,
                    previous_mapping=previous_mapping,
                    current_cluster_count=artifact.n_clusters,
                )

            artifact = self._attach_cluster_mapping(
                artifact,
                cluster_mapping,
            )

            self.registry.save(artifact)

            diagnostics.append(
                {
                    "model_version": model_version,
                    "training_cutoff": origin.training_cutoff.isoformat(),
                    **model_diagnostics,
                }
            )

            if index == 0:
                baseline = history.loc[
                    history["snapshot_date"]
                    <= origin.training_cutoff
                ].copy()

                assigned_history.append(
                    self._assign_snapshots(
                        baseline,
                        artifact,
                        cluster_mapping,
                    )
                )

            next_cutoff = (
                origin.next_training_cutoff
                if origin.next_training_cutoff is not None
                else history["snapshot_date"].max()
            )

            forward_data = history.loc[
                (history["snapshot_date"] > origin.training_cutoff)
                & (history["snapshot_date"] <= next_cutoff)
            ].copy()

            if not forward_data.empty:
                assigned_history.append(
                    self._assign_snapshots(
                        forward_data,
                        artifact,
                        cluster_mapping,
                    )
                )

            previous_artifact = artifact

        behavioral_history = (
            pd.concat(assigned_history, ignore_index=True)
            .sort_values(
                ["snapshot_date", "symbol"],
                ignore_index=True,
            )
        )

        return WalkForwardResult(
            behavioral_history=behavioral_history,
            model_versions=self.registry.list_versions(),
            diagnostics=diagnostics,
            alignment_results=alignment_results,
        )

    def _align_models(
        self,
        previous_artifact: ClusteringModelArtifact,
        current_artifact: ClusteringModelArtifact,
        history: pd.DataFrame,
        cutoff: pd.Timestamp,
    ) -> ClusterAlignmentResult:
        comparison_data = history.loc[
            history["snapshot_date"] <= cutoff
        ].copy()

        features = comparison_data.loc[
            :, list(self.feature_columns)
        ]

        previous_labels = previous_artifact.predict(features)
        current_labels = current_artifact.predict(features)

        previous_profiles = build_cluster_profiles(
            features,
            previous_labels,
        )

        current_profiles = build_cluster_profiles(
            features,
            current_labels,
        )

        previous_features = self._cluster_feature_frames(
            features,
            previous_labels,
        )

        current_features = self._cluster_feature_frames(
            features,
            current_labels,
        )

        latest_date = comparison_data["snapshot_date"].max()

        latest_data = comparison_data.loc[
            comparison_data["snapshot_date"] == latest_date
        ]

        latest_features = latest_data.loc[
            :, list(self.feature_columns)
        ]

        previous_latest_labels = previous_artifact.predict(
            latest_features
        )

        current_latest_labels = current_artifact.predict(
            latest_features
        )

        previous_assignments = self._build_assignments(
            latest_data,
            previous_latest_labels,
        )

        current_assignments = self._build_assignments(
            latest_data,
            current_latest_labels,
        )

        reference_scale = pd.Series(
            current_artifact.scaler.scale_,
            index=self.feature_columns,
        )

        return self.alignment_engine.align(
            previous_profiles=previous_profiles,
            current_profiles=current_profiles,
            reference_scale=reference_scale,
            previous_assignments=previous_assignments,
            current_assignments=current_assignments,
            threshold_estimator=self.threshold_estimator,
            previous_features=previous_features,
            current_features=current_features,
        )

    @staticmethod
    def _cluster_feature_frames(
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> dict[int, pd.DataFrame]:
        result: dict[int, pd.DataFrame] = {}

        for cluster_id in sorted(labels.unique()):
            result[int(cluster_id)] = features.loc[
                labels == cluster_id
            ].copy()

        return result

    @staticmethod
    def _build_assignments(
        data: pd.DataFrame,
        labels: pd.Series,
    ) -> list[AssetClusterAssignment]:
        return [
            AssetClusterAssignment(
                asset=str(symbol),
                cluster_id=int(cluster_id),
            )
            for symbol, cluster_id in zip(
                data["symbol"],
                labels,
                strict=True,
            )
        ]

    @staticmethod
    def _read_mapping(
        artifact: ClusteringModelArtifact,
    ) -> dict[int, int]:
        mapping = artifact.metadata.get(
            "cluster_id_mapping",
            {},
        )

        return {
            int(raw_id): int(canonical_id)
            for raw_id, canonical_id in mapping.items()
        }

    @staticmethod
    def _build_cluster_mapping(
        alignment: ClusterAlignmentResult,
        previous_mapping: dict[int, int],
        current_cluster_count: int,
    ) -> dict[int, int]:
        mapping: dict[int, int] = {}

        previous_canonical_ids = list(previous_mapping.values())

        if len(previous_canonical_ids) != len(set(previous_canonical_ids)):
            raise ValueError("Previous canonical cluster IDs must be unique.")

        next_id = max(previous_canonical_ids, default=-1) + 1

        for match in alignment.matches:
            if (
                match.old_cluster_id is None
                or match.new_cluster_id is None
            ):
                continue

            if match.match_status not in {"matched", "conditional"}:
                continue

            old_id = int(match.old_cluster_id)
            new_id = int(match.new_cluster_id)

            if old_id not in previous_mapping:
                raise ValueError(
                    f"Previous cluster ID {old_id} is missing from previous mapping."
                )

            if new_id in mapping:
                raise ValueError(
                    f"Current cluster ID {new_id} has multiple canonical mappings."
                )

            mapping[new_id] = previous_mapping[old_id]

        for raw_id in range(current_cluster_count):
            if raw_id not in mapping:
                mapping[raw_id] = next_id
                next_id += 1

        return mapping

    @staticmethod
    def _attach_cluster_mapping(
        artifact: ClusteringModelArtifact,
        mapping: dict[int, int],
    ) -> ClusteringModelArtifact:
        return replace(
            artifact,
            metadata={
                **artifact.metadata,
                "cluster_id_mapping": {
                    str(raw_id): canonical_id
                    for raw_id, canonical_id in mapping.items()
                },
            },
        )

    def _assign_snapshots(
        self,
        data: pd.DataFrame,
        artifact: ClusteringModelArtifact,
        cluster_mapping: dict[int, int],
    ) -> pd.DataFrame:
        result = data.copy()

        features = result.loc[
            :, list(self.feature_columns)
        ]

        raw_labels = artifact.predict(features)

        missing_ids = set(raw_labels.unique()) - set(
            cluster_mapping
        )

        if missing_ids:
            raise RuntimeError(
                f"Missing cluster mappings: {sorted(missing_ids)}"
            )

        result["raw_cluster_id"] = raw_labels.to_numpy()

        result["cluster_id"] = [
            cluster_mapping[int(label)]
            for label in raw_labels
        ]

        result["model_version"] = artifact.model_version

        return result

    def _validate_snapshots(
        self,
        snapshots: pd.DataFrame,
    ) -> pd.DataFrame:
        required_columns = {
            "snapshot_date",
            "symbol",
            *self.feature_columns,
        }

        missing = required_columns - set(snapshots.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        data = snapshots.copy()

        data["snapshot_date"] = pd.to_datetime(
            data["snapshot_date"]
        )

        if data["snapshot_date"].isna().any():
            raise ValueError("Snapshot dates cannot be missing.")

        if data["symbol"].isna().any():
            raise ValueError("Asset symbols cannot be missing.")

        if data.duplicated(
            ["snapshot_date", "symbol"]
        ).any():
            raise ValueError(
                "Duplicate asset snapshots detected."
            )

        values = data.loc[
            :, list(self.feature_columns)
        ].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                "Snapshot features must be finite."
            )

        return data.sort_values(
            ["snapshot_date", "symbol"],
            ignore_index=True,
        )
