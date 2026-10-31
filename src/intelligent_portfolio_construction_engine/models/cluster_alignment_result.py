from dataclasses import dataclass, field

from intelligent_portfolio_construction_engine.models.match_decision import (
    MatchDecision,
)


@dataclass(frozen=True)
class ClusterMatch:
    old_cluster_id: int | None
    new_cluster_id: int | None
    match_cost: float | None
    match_status: str


@dataclass(frozen=True)
class ClusterStructuralEvent:
    event_type: str
    old_cluster_ids: tuple[int, ...]
    new_cluster_ids: tuple[int, ...]
    shared_assets: int


@dataclass(frozen=True)
class ClusterAlignmentResult:
    matches: list[ClusterMatch]
    decisions: list[MatchDecision] = field(default_factory=list)
    structural_events: list[ClusterStructuralEvent] = field(
        default_factory=list
    )