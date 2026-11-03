from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from intelligent_portfolio_construction_engine.clustering.match_threshold import (
    BootstrapThresholdEstimator,
)
from intelligent_portfolio_construction_engine.clustering.model_version_registry import (
    ModelVersionRegistry,
)
from intelligent_portfolio_construction_engine.clustering.walk_forward_engine import (
    WalkForwardEngine,
)
from intelligent_portfolio_construction_engine.clustering.walk_forward_schedule import (
    WalkForwardSchedule,
)
from intelligent_portfolio_construction_engine.clustering.walk_forward_trainer import (
    WalkForwardModelTrainer,
)
from intelligent_portfolio_construction_engine.clustering.cluster_profiling import (
    FEATURE_COLUMNS,
)
from intelligent_portfolio_construction_engine.models.cluster_alignment_result import (
    ClusterAlignmentResult,
    ClusterMatch,
)
from intelligent_portfolio_construction_engine.models.clustering_model_artifact import (
    ClusteringModelArtifact,
)


class DummyScaler:
    def transform(self, data):
        return np.asarray(data, dtype=float)


class DummyEstimator:
    def predict(self, data):
        return np.zeros(len(data), dtype=int)


def make_snapshot_dates(start="2020-01-01", periods=260):
    return pd.Series(
        pd.date_range(start=start, periods=periods, freq="W")
    )


def make_snapshots(periods=260, symbols=12, seed=42):
    rng = np.random.default_rng(seed)
    dates = make_snapshot_dates(periods=periods)

    rows = []

    for date in dates:
        for symbol_index in range(symbols):
            row = {
                "snapshot_date": date,
                "symbol": f"ASSET_{symbol_index:02d}",
            }

            for feature_index, feature in enumerate(FEATURE_COLUMNS):
                row[feature] = (
                    rng.normal()
                    + symbol_index * 0.15
                    + feature_index * 0.05
                )

            rows.append(row)

    return pd.DataFrame(rows)


def make_artifact(
    model_version="wf-v1",
    feature_columns=("feature_a", "feature_b"),
):
    return ClusteringModelArtifact(
        estimator=DummyEstimator(),
        scaler=DummyScaler(),
        feature_columns=feature_columns,
        model_name="dummy",
        n_clusters=2,
        training_cutoff=pd.Timestamp("2022-01-01"),
        model_version=model_version,
        metadata={"snapshot_count": 100},
    )


class TestWalkForwardSchedule:

    def test_generates_training_origins(self):
        schedule = WalkForwardSchedule(
            retraining_interval_weeks=52,
            minimum_training_weeks=104,
        )

        origins = schedule.generate(make_snapshot_dates())

        assert len(origins) > 0

    def test_versions_are_sequential(self):
        schedule = WalkForwardSchedule()
        origins = schedule.generate(make_snapshot_dates())

        assert [origin.version for origin in origins] == list(
            range(1, len(origins) + 1)
        )

    def test_cutoffs_are_available_dates(self):
        dates = make_snapshot_dates()
        schedule = WalkForwardSchedule()

        origins = schedule.generate(dates)
        available_dates = set(pd.to_datetime(dates))

        for origin in origins:
            assert origin.training_cutoff in available_dates

    def test_cutoffs_are_increasing(self):
        schedule = WalkForwardSchedule()
        origins = schedule.generate(make_snapshot_dates())

        cutoffs = [origin.training_cutoff for origin in origins]

        assert cutoffs == sorted(cutoffs)
        assert len(cutoffs) == len(set(cutoffs))

    def test_accepts_reordered_feature_columns(self):
        artifact = make_artifact()

        data = pd.DataFrame(
            {
                "feature_b": [1.0],
                "feature_a": [2.0],
            }
        )

        predictions = artifact.predict(data)

        assert len(predictions) == 1


class TestClusteringModelArtifact:

    def test_predict_returns_expected_number_of_labels(self):
        artifact = make_artifact()

        data = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0],
                "feature_b": [4.0, 5.0, 6.0],
            }
        )

        predictions = artifact.predict(data)

        assert len(predictions) == len(data)

    def test_rejects_missing_feature(self):
        artifact = make_artifact()

        data = pd.DataFrame({"feature_a": [1.0]})

        with pytest.raises(ValueError):
            artifact.predict(data)

    def test_rejects_extra_feature(self):
        artifact = make_artifact()

        data = pd.DataFrame(
            {
                "feature_a": [1.0],
                "feature_b": [2.0],
                "extra": [3.0],
            }
        )

        with pytest.raises(ValueError):
            artifact.predict(data)

   


class TestWalkForwardModelTrainer:

    @pytest.fixture
    def snapshots(self):
        return make_snapshots(periods=180, symbols=8)

    @pytest.fixture
    def trainer(self):
        return WalkForwardModelTrainer(
            feature_columns=tuple(FEATURE_COLUMNS),
            min_k=2,
            max_k=3,
        )

    def test_training_uses_only_data_before_cutoff(
    self,
    trainer,
    snapshots,
):
        cutoff = snapshots["snapshot_date"].drop_duplicates().iloc[129]

        artifact, diagnostics = trainer.fit(
            snapshots=snapshots,
            training_cutoff=cutoff,
            model_version="wf-v1",
        )

        assert artifact.training_cutoff == cutoff
        assert artifact.metadata["snapshot_count"] == len(
            snapshots.loc[snapshots["snapshot_date"] <= cutoff]
        )
        assert isinstance(diagnostics, dict)

    def test_future_data_does_not_change_scaler(
        self,
        trainer,
        snapshots,
    ):
        cutoff = snapshots["snapshot_date"].drop_duplicates().iloc[129]

        first_artifact, _ = trainer.fit(
            snapshots=snapshots,
            training_cutoff=cutoff,
            model_version="wf-v1",
        )

        modified = snapshots.copy()
        future_mask = modified["snapshot_date"] > cutoff

        modified.loc[future_mask, list(FEATURE_COLUMNS)] += 1000

        second_artifact, _ = trainer.fit(
            snapshots=modified,
            training_cutoff=cutoff,
            model_version="wf-v1",
        )

        np.testing.assert_allclose(
            first_artifact.scaler.center_,
            second_artifact.scaler.center_,
        )

        np.testing.assert_allclose(
            first_artifact.scaler.scale_,
            second_artifact.scaler.scale_,
        )

    def test_rejects_missing_feature_values(
        self,
        trainer,
        snapshots,
    ):
        snapshots.loc[0, FEATURE_COLUMNS[0]] = np.nan

        cutoff = snapshots["snapshot_date"].drop_duplicates().iloc[129]

        with pytest.raises(ValueError):
            trainer.fit(
                snapshots=snapshots,
                training_cutoff=cutoff,
                model_version="wf-v1",
            )

    


class TestModelVersionRegistry:

    @pytest.fixture
    def registry(self, tmp_path):
        return ModelVersionRegistry(root_dir=tmp_path)

    def test_saves_and_loads_artifact(self, registry):
        artifact = make_artifact()

        registry.save(artifact)
        loaded = registry.load("wf-v1")

        assert loaded.model_version == artifact.model_version
        assert loaded.n_clusters == artifact.n_clusters

    def test_loaded_artifact_can_predict(self, registry):
        registry.save(make_artifact())

        loaded = registry.load("wf-v1")

        data = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0],
                "feature_b": [3.0, 4.0],
            }
        )

        predictions = loaded.predict(data)

        assert len(predictions) == 2

    def test_rejects_duplicate_version(self, registry):
        registry.save(make_artifact())

        with pytest.raises(FileExistsError):
            registry.save(make_artifact())

    def test_lists_saved_versions(self, registry):
        registry.save(make_artifact())

        versions = registry.list_versions()

        assert len(versions) == 1
        assert versions[0]["version"] == "wf-v1"

    def test_returns_latest_version(self, registry):
        registry.save(make_artifact("wf-v1"))

        assert registry.get_latest_version() == "wf-v1"

    def test_empty_registry_has_no_latest_version(self, registry):
        assert registry.get_latest_version() is None

    def test_unknown_version_raises_error(self, registry):
        with pytest.raises((KeyError, ValueError, FileNotFoundError)):
            registry.load("wf-unknown")


class TestCanonicalClusterMapping:

    @pytest.fixture
    def engine(self, tmp_path):
        return WalkForwardEngine(
            registry=ModelVersionRegistry(root_dir=tmp_path),
            threshold_estimator=BootstrapThresholdEstimator(
                n_bootstrap=100,
                random_state=42,
            ),
            min_k=2,
            max_k=3,
        )

    def test_matched_cluster_preserves_canonical_id(self, engine):
        alignment = ClusterAlignmentResult(
            matches=[
                ClusterMatch(
                    old_cluster_id=0,
                    new_cluster_id=1,
                    match_cost=0.1,
                    match_status="matched",
                )
            ]
        )

        mapping = engine._build_cluster_mapping(
            alignment=alignment,
            previous_mapping={0: 10, 1: 11},
            current_cluster_count=2,
        )

        assert mapping[1] == 10

    def test_new_cluster_gets_new_canonical_id(self, engine):
        alignment = ClusterAlignmentResult(
            matches=[
                ClusterMatch(
                    old_cluster_id=0,
                    new_cluster_id=0,
                    match_cost=0.1,
                    match_status="matched",
                )
            ]
        )

        mapping = engine._build_cluster_mapping(
            alignment=alignment,
            previous_mapping={0: 10},
            current_cluster_count=2,
        )

        assert mapping[0] == 10
        assert mapping[1] == 11

    def test_conditional_match_preserves_id(self, engine):
        alignment = ClusterAlignmentResult(
            matches=[
                ClusterMatch(
                    old_cluster_id=0,
                    new_cluster_id=1,
                    match_cost=0.4,
                    match_status="conditional",
                )
            ]
        )

        mapping = engine._build_cluster_mapping(
            alignment=alignment,
            previous_mapping={0: 7},
            current_cluster_count=2,
        )

        assert mapping[1] == 7

    def test_unmatched_cluster_gets_new_id(self, engine):
        alignment = ClusterAlignmentResult(
            matches=[
                ClusterMatch(
                    old_cluster_id=0,
                    new_cluster_id=None,
                    match_cost=None,
                    match_status="retired",
                )
            ]
        )

        mapping = engine._build_cluster_mapping(
            alignment=alignment,
            previous_mapping={0: 5},
            current_cluster_count=1,
        )

        assert mapping == {0: 6}

   
    def test_rejects_duplicate_current_mapping(self, engine):
        alignment = ClusterAlignmentResult(
            matches=[
                ClusterMatch(0, 0, 0.1, "matched"),
                ClusterMatch(1, 0, 0.2, "matched"),
            ]
        )

        with pytest.raises(ValueError):
            engine._build_cluster_mapping(
                alignment=alignment,
                previous_mapping={0: 5, 1: 6},
                current_cluster_count=2,
            )

    def test_rejects_missing_previous_mapping(self, engine):
        alignment = ClusterAlignmentResult(
            matches=[
                ClusterMatch(5, 0, 0.1, "matched"),
            ]
        )

        with pytest.raises(ValueError):
            engine._build_cluster_mapping(
                alignment=alignment,
                previous_mapping={0: 5},
                current_cluster_count=1,
            )


class TestWalkForwardEngineIntegration:

    @pytest.fixture
    def snapshots(self):
        return make_snapshots(periods=260, symbols=12)

    @pytest.fixture
    def engine(self, tmp_path):
        registry = ModelVersionRegistry(root_dir=tmp_path)

        return WalkForwardEngine(
            registry=registry,
            threshold_estimator=BootstrapThresholdEstimator(
                n_bootstrap=100,
                random_state=42,
            ),
            min_k=2,
            max_k=3,
            retraining_interval_weeks=52,
            minimum_training_weeks=104,
        )

    def test_returns_walk_forward_result(self, engine, snapshots):
        result = engine.run(snapshots)

        assert not result.behavioral_history.empty
        assert len(result.model_versions) > 0

    def test_output_contains_required_columns(self, engine, snapshots):
        result = engine.run(snapshots)

        required = {
            "snapshot_date",
            "symbol",
            "raw_cluster_id",
            "cluster_id",
            "model_version",
        }

        assert required.issubset(result.behavioral_history.columns)

    def test_output_has_no_duplicate_asset_snapshots(
        self,
        engine,
        snapshots,
    ):
        result = engine.run(snapshots)
        history = result.behavioral_history

        assert not history.duplicated(
            ["snapshot_date", "symbol"]
        ).any()

    def test_output_is_sorted(self, engine, snapshots):
        result = engine.run(snapshots)
        history = result.behavioral_history

        expected = history.sort_values(
            ["snapshot_date", "symbol"],
            ignore_index=True,
        )

        pd.testing.assert_frame_equal(history, expected)

    def test_every_assignment_references_a_saved_model(
        self,
        engine,
        snapshots,
    ):
        result = engine.run(snapshots)

        saved_versions = {
            item["version"] for item in result.model_versions
        }

        assigned_versions = set(
            result.behavioral_history["model_version"].unique()
        )

        assert assigned_versions.issubset(saved_versions)

    def test_future_snapshots_use_models_trained_before_them(
        self,
        engine,
        snapshots,
    ):
        result = engine.run(snapshots)
        history = result.behavioral_history

        for version in result.model_versions:
            model_version = version["version"]
            cutoff = pd.Timestamp(version["training_cutoff"])

            assigned_dates = history.loc[
                history["model_version"] == model_version,
                "snapshot_date",
            ]

            if model_version == "wf-v1":
                assert (assigned_dates <= cutoff).any()
            else:
                assert (assigned_dates > cutoff).all()

    def test_diagnostics_are_returned(self, engine, snapshots):
        result = engine.run(snapshots)

        assert isinstance(result.diagnostics, list)
        assert len(result.diagnostics) == len(result.model_versions)

    def test_alignment_results_are_returned(self, engine, snapshots):
        result = engine.run(snapshots)

        assert isinstance(result.alignment_results, dict)

    def test_rejects_missing_required_column(self, engine, snapshots):
        invalid = snapshots.drop(columns=["symbol"])

        with pytest.raises(ValueError):
            engine.run(invalid)

    def test_rejects_duplicate_asset_snapshots(self, engine, snapshots):
        invalid = pd.concat(
            [snapshots, snapshots.iloc[[0]]],
            ignore_index=True,
        )

        with pytest.raises(ValueError):
            engine.run(invalid)

    def test_rejects_missing_feature_values(self, engine, snapshots):
        invalid = snapshots.copy()
        invalid.loc[0, FEATURE_COLUMNS[0]] = np.nan

        with pytest.raises(ValueError):
            engine.run(invalid)