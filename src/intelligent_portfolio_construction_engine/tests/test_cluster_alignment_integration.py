import numpy as np
import pandas as pd
import pytest

from intelligent_portfolio_construction_engine.clustering.cluster_alignment import (
    ClusterAlignmentEngine,
)
from intelligent_portfolio_construction_engine.clustering.cluster_profiling import (
    FEATURE_COLUMNS,
)
from intelligent_portfolio_construction_engine.models.asset_cluster_assignment import (
    AssetClusterAssignment,
)
from intelligent_portfolio_construction_engine.models.cluster_profile import (
    ClusterProfile,
)
from intelligent_portfolio_construction_engine.models.match_threshold import (
    MatchThreshold,
)


class FakeThresholdEstimator:
    def __init__(
        self,
        threshold_95: float = 1.0,
        threshold_99: float = 2.0,
    ) -> None:
        self.threshold_95 = threshold_95
        self.threshold_99 = threshold_99

    def estimate(
        self,
        previous_features: pd.DataFrame,
        current_features: pd.DataFrame,
        reference_scale: pd.Series,
    ) -> MatchThreshold:
        return MatchThreshold(
            threshold_95=self.threshold_95,
            threshold_99=self.threshold_99,
        )


@pytest.fixture
def reference_scale() -> pd.Series:
    return pd.Series(
        {feature: 1.0 for feature in FEATURE_COLUMNS}
    )


@pytest.fixture
def threshold_estimator() -> FakeThresholdEstimator:
    return FakeThresholdEstimator()


def make_profile(
    cluster_id: int,
    value: float,
) -> ClusterProfile:
    return ClusterProfile(
        cluster_id=cluster_id,
        asset_count=2,
        centroid={feature: value for feature in FEATURE_COLUMNS},
        median_profile={feature: value for feature in FEATURE_COLUMNS},
    )


def make_features(value: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame(
        [{feature: value for feature in FEATURE_COLUMNS}]
    )


def make_assignments(
    assignments: list[tuple[str, int]],
) -> list[AssetClusterAssignment]:
    return [
        AssetClusterAssignment(asset=asset, cluster_id=cluster_id)
        for asset, cluster_id in assignments
    ]


def run_alignment(
    previous_profiles,
    current_profiles,
    reference_scale,
    threshold_estimator,
    previous_assignments=None,
    current_assignments=None,
):
    previous_features = {
        profile.cluster_id: make_features()
        for profile in previous_profiles
    }

    current_features = {
        profile.cluster_id: make_features()
        for profile in current_profiles
    }

    engine = ClusterAlignmentEngine()

    return engine.align(
        previous_profiles=previous_profiles,
        current_profiles=current_profiles,
        reference_scale=reference_scale,
        previous_assignments=previous_assignments,
        current_assignments=current_assignments,
        threshold_estimator=threshold_estimator,
        previous_features=previous_features,
        current_features=current_features,
    )


def test_successful_match(
    reference_scale,
    threshold_estimator,
):
    result = run_alignment(
        previous_profiles=[make_profile(0, 0.0)],
        current_profiles=[make_profile(1, 0.1)],
        reference_scale=reference_scale,
        threshold_estimator=threshold_estimator,
    )

    assert len(result.matches) == 1
    assert result.matches[0].match_status == "matched"
    assert result.matches[0].old_cluster_id == 0
    assert result.matches[0].new_cluster_id == 1

    assert len(result.decisions) == 1
    assert result.decisions[0].confidence_level == "within_95"


def test_conditional_match(reference_scale):
    estimator = FakeThresholdEstimator(
        threshold_95=1.0,
        threshold_99=2.0,
    )

    result = run_alignment(
        previous_profiles=[make_profile(0, 0.0)],
        current_profiles=[make_profile(1, 0.5)],
        reference_scale=reference_scale,
        threshold_estimator=estimator,
    )

    assert len(result.matches) == 1
    assert result.matches[0].match_status == "conditional"
    assert result.decisions[0].confidence_level == "within_99"

def test_rejected_match_creates_new_and_retired(
    reference_scale,
):
    estimator = FakeThresholdEstimator(
        threshold_95=0.1,
        threshold_99=0.2,
    )

    result = run_alignment(
        previous_profiles=[make_profile(0, 0.0)],
        current_profiles=[make_profile(1, 1.0)],
        reference_scale=reference_scale,
        threshold_estimator=estimator,
    )

    statuses = {match.match_status for match in result.matches}

    assert statuses == {"new", "retired"}
    assert result.decisions == []


def test_split_detection(
    reference_scale,
    threshold_estimator,
):
    previous_assignments = make_assignments(
        [("AAPL", 0), ("MSFT", 0)]
    )

    current_assignments = make_assignments(
        [("AAPL", 1), ("MSFT", 2)]
    )

    result = run_alignment(
        previous_profiles=[make_profile(0, 0.0)],
        current_profiles=[
            make_profile(1, 0.1),
            make_profile(2, 0.2),
        ],
        reference_scale=reference_scale,
        threshold_estimator=threshold_estimator,
        previous_assignments=previous_assignments,
        current_assignments=current_assignments,
    )

    assert len(result.structural_events) == 1

    event = result.structural_events[0]

    assert event.event_type == "split"
    assert event.old_cluster_ids == (0,)
    assert event.new_cluster_ids == (1, 2)
    assert event.shared_assets == 2


def test_merge_detection(
    reference_scale,
    threshold_estimator,
):
    previous_assignments = make_assignments(
        [("AAPL", 0), ("MSFT", 1)]
    )

    current_assignments = make_assignments(
        [("AAPL", 2), ("MSFT", 2)]
    )

    result = run_alignment(
        previous_profiles=[
            make_profile(0, 0.0),
            make_profile(1, 0.1),
        ],
        current_profiles=[make_profile(2, 0.05)],
        reference_scale=reference_scale,
        threshold_estimator=threshold_estimator,
        previous_assignments=previous_assignments,
        current_assignments=current_assignments,
    )

    assert len(result.structural_events) == 1

    event = result.structural_events[0]

    assert event.event_type == "merge"
    assert event.old_cluster_ids == (0, 1)
    assert event.new_cluster_ids == (2,)
    assert event.shared_assets == 2
