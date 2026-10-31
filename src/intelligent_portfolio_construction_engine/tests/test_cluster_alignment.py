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


@pytest.fixture
def engine():
    return ClusterAlignmentEngine()


@pytest.fixture
def reference_scale():
    return pd.Series({feature: 1.0 for feature in FEATURE_COLUMNS})


def make_profile(cluster_id, values):
    return ClusterProfile(
        cluster_id=cluster_id,
        asset_count=10,
        centroid=dict(zip(FEATURE_COLUMNS, values, strict=True)),
        median_profile=dict(zip(FEATURE_COLUMNS, values, strict=True)),
    )


def make_assignment(asset, cluster_id):
    return AssetClusterAssignment(
        asset=asset,
        cluster_id=cluster_id,
    )


def test_align_reordered_clusters(engine, reference_scale):
    previous = [
        make_profile(0, [1, 1, 1, 1, 1, 1, 1]),
        make_profile(1, [5, 5, 5, 5, 5, 5, 5]),
    ]

    current = [
        make_profile(0, [5.1, 5, 5, 5, 5, 5, 5]),
        make_profile(1, [1.1, 1, 1, 1, 1, 1, 1]),
    ]

    assignments = [
        make_assignment("A", 0),
        make_assignment("B", 1),
    ]

    result = engine.align(
        previous,
        current,
        reference_scale,
        assignments,
        assignments,
    )

    mapping = {
        item.old_cluster_id: item.new_cluster_id
        for item in result.matches
        if item.match_status == "matched"
    }

    assert mapping == {0: 1, 1: 0}
    assert result.structural_events == []


def test_detect_split_candidate(engine, reference_scale):
    previous = [
        make_profile(0, [1, 1, 1, 1, 1, 1, 1]),
    ]

    current = [
        make_profile(2, [1.1, 1, 1, 1, 1, 1, 1]),
        make_profile(3, [2, 2, 2, 2, 2, 2, 2]),
    ]

    previous_assignments = [
        make_assignment("A", 0),
        make_assignment("B", 0),
        make_assignment("C", 0),
        make_assignment("D", 0),
    ]

    current_assignments = [
        make_assignment("A", 2),
        make_assignment("B", 2),
        make_assignment("C", 3),
        make_assignment("D", 3),
    ]

    result = engine.align(
        previous,
        current,
        reference_scale,
        previous_assignments,
        current_assignments,
    )

    assert any(
        event.event_type == "split_candidate"
        and event.old_cluster_ids == (0,)
        and event.new_cluster_ids == (2, 3)
        for event in result.structural_events
    )


def test_detect_merge_candidate(engine, reference_scale):
    previous = [
        make_profile(0, [1, 1, 1, 1, 1, 1, 1]),
        make_profile(1, [2, 2, 2, 2, 2, 2, 2]),
    ]

    current = [
        make_profile(2, [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]),
    ]

    previous_assignments = [
        make_assignment("A", 0),
        make_assignment("B", 0),
        make_assignment("C", 1),
        make_assignment("D", 1),
    ]

    current_assignments = [
        make_assignment("A", 2),
        make_assignment("B", 2),
        make_assignment("C", 2),
        make_assignment("D", 2),
    ]

    result = engine.align(
        previous,
        current,
        reference_scale,
        previous_assignments,
        current_assignments,
    )

    assert any(
        event.event_type == "merge_candidate"
        and event.old_cluster_ids == (0, 1)
        and event.new_cluster_ids == (2,)
        for event in result.structural_events
    )


def test_detect_new_and_retired_clusters(engine, reference_scale):
    previous = [
        make_profile(0, [1, 1, 1, 1, 1, 1, 1]),
        make_profile(1, [5, 5, 5, 5, 5, 5, 5]),
    ]

    current = [
        make_profile(2, [1.1, 1, 1, 1, 1, 1, 1]),
        make_profile(3, [9, 9, 9, 9, 9, 9, 9]),
    ]

    previous_assignments = [
        make_assignment("A", 0),
        make_assignment("B", 1),
    ]

    current_assignments = [
        make_assignment("A", 2),
        make_assignment("B", 3),
    ]

    result = engine.align(
        previous,
        current,
        reference_scale,
        previous_assignments,
        current_assignments,
    )

    statuses = {item.match_status for item in result.matches}

    assert "new" in statuses
    assert "retired" in statuses


def test_empty_previous_profiles(engine, reference_scale):
    current = [
        make_profile(0, [1, 1, 1, 1, 1, 1, 1]),
    ]

    result = engine.align(
        [],
        current,
        reference_scale,
        [],
        [make_assignment("A", 0)],
    )

    assert len(result.matches) == 1
    assert result.matches[0].match_status == "new"


def test_empty_current_profiles(engine, reference_scale):
    previous = [
        make_profile(0, [1, 1, 1, 1, 1, 1, 1]),
    ]

    result = engine.align(
        previous,
        [],
        reference_scale,
        [make_assignment("A", 0)],
        [],
    )

    assert len(result.matches) == 1
    assert result.matches[0].match_status == "retired"


def test_duplicate_asset_assignment_raises_error(engine, reference_scale):
    assignments = [
        make_assignment("A", 0),
        make_assignment("A", 1),
    ]

    with pytest.raises(ValueError):
        engine.align(
            [make_profile(0, [1] * 7)],
            [make_profile(1, [1] * 7)],
            reference_scale,
            assignments,
            [make_assignment("A", 1)],
        )


def test_invalid_reference_scale_raises_error(engine):
    previous = [make_profile(0, [1] * 7)]
    current = [make_profile(1, [1] * 7)]

    invalid_scale = pd.Series({"volatility": 1.0})

    with pytest.raises(ValueError):
        engine.align(
            previous,
            current,
            invalid_scale,
            [make_assignment("A", 0)],
            [make_assignment("A", 1)],
        )


def test_no_shared_assets_produces_no_structural_events(
    engine,
    reference_scale,
):
    previous = [make_profile(0, [1] * 7)]
    current = [make_profile(1, [2] * 7)]

    result = engine.align(
        previous,
        current,
        reference_scale,
        [make_assignment("A", 0)],
        [make_assignment("B", 1)],
    )

    assert result.structural_events == []