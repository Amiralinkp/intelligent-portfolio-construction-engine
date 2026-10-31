import pytest

from intelligent_portfolio_construction_engine.models.match_decision import (
    MatchDecision,
)


def test_match_decision_creation():
    decision = MatchDecision(
        old_cluster_id=1,
        new_cluster_id=2,
        match_cost=0.5,
        threshold_95=0.8,
        threshold_99=1.2,
        match_status="matched",
        confidence_level="high",
    )

    assert decision.old_cluster_id == 1
    assert decision.new_cluster_id == 2
    assert decision.match_cost == 0.5
    assert decision.threshold_95 == 0.8
    assert decision.threshold_99 == 1.2
    assert decision.match_status == "matched"
    assert decision.confidence_level == "high"


def test_match_decision_is_immutable():
    decision = MatchDecision(
        old_cluster_id=1,
        new_cluster_id=2,
        match_cost=0.5,
        threshold_95=0.8,
        threshold_99=1.2,
        match_status="conditional",
        confidence_level="medium",
    )

    with pytest.raises(AttributeError):
        decision.match_status = "rejected"