from intelligent_portfolio_construction_engine.clustering.cluster_membership import (
    ClusterMembershipAnalyzer,
)
from intelligent_portfolio_construction_engine.models.asset_cluster_assignment import (
    AssetClusterAssignment,
)


def make_assignment(asset, cluster_id):
    return AssetClusterAssignment(
        asset=asset,
        cluster_id=cluster_id,
    )


def test_basic_membership_overlap():
    analyzer = ClusterMembershipAnalyzer()

    previous = [
        make_assignment("A", 0),
        make_assignment("B", 0),
        make_assignment("C", 1),
    ]

    current = [
        make_assignment("A", 2),
        make_assignment("B", 2),
        make_assignment("C", 1),
    ]

    result = analyzer.analyze(previous, current)

    assert len(result.overlaps) == 2

    overlap = next(
        item
        for item in result.overlaps
        if item.old_cluster_id == 0 and item.new_cluster_id == 2
    )

    assert overlap.shared_assets == 2
    assert overlap.old_coverage == 1.0
    assert overlap.new_coverage == 1.0


def test_split_membership():
    analyzer = ClusterMembershipAnalyzer()

    previous = [
        make_assignment("A", 0),
        make_assignment("B", 0),
        make_assignment("C", 0),
        make_assignment("D", 1),
    ]

    current = [
        make_assignment("A", 2),
        make_assignment("B", 2),
        make_assignment("C", 3),
        make_assignment("D", 1),
    ]

    result = analyzer.analyze(previous, current)

    split_overlaps = [
        item
        for item in result.overlaps
        if item.old_cluster_id == 0
    ]

    assert len(split_overlaps) == 2
    assert {item.new_cluster_id for item in split_overlaps} == {2, 3}


def test_merge_membership():
    analyzer = ClusterMembershipAnalyzer()

    previous = [
        make_assignment("A", 0),
        make_assignment("B", 0),
        make_assignment("C", 1),
        make_assignment("D", 1),
    ]

    current = [
        make_assignment("A", 2),
        make_assignment("B", 2),
        make_assignment("C", 2),
        make_assignment("D", 2),
    ]

    result = analyzer.analyze(previous, current)

    merge_overlaps = [
        item
        for item in result.overlaps
        if item.new_cluster_id == 2
    ]

    assert len(merge_overlaps) == 2
    assert {item.old_cluster_id for item in merge_overlaps} == {0, 1}


def test_no_shared_assets():
    analyzer = ClusterMembershipAnalyzer()

    previous = [
        make_assignment("A", 0),
        make_assignment("B", 0),
    ]

    current = [
        make_assignment("C", 1),
        make_assignment("D", 1),
    ]

    result = analyzer.analyze(previous, current)

    assert result.overlaps == []


def test_empty_assignments():
    analyzer = ClusterMembershipAnalyzer()

    result = analyzer.analyze([], [])

    assert result.overlaps == []


def test_partial_asset_overlap():
    analyzer = ClusterMembershipAnalyzer()

    previous = [
        make_assignment("A", 0),
        make_assignment("B", 0),
        make_assignment("C", 0),
    ]

    current = [
        make_assignment("A", 1),
        make_assignment("B", 1),
        make_assignment("D", 1),
    ]

    result = analyzer.analyze(previous, current)

    assert len(result.overlaps) == 1

    overlap = result.overlaps[0]

    assert overlap.shared_assets == 2
    assert overlap.old_coverage == 2 / 3
    assert overlap.new_coverage == 2 / 3