from dataclasses import dataclass


@dataclass(frozen=True)
class MatchDecision:
    old_cluster_id: int
    new_cluster_id: int
    match_cost: float
    threshold_95: float
    threshold_99: float
    match_status: str
    confidence_level: str