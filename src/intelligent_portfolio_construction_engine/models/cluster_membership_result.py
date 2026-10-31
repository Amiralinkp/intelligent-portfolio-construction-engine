from dataclasses import dataclass


@dataclass(frozen=True)
class ClusterOverlap:
    old_cluster_id: int
    new_cluster_id: int
    shared_assets: int
    old_coverage: float
    new_coverage: float


@dataclass(frozen=True)
class ClusterMembershipResult:
    overlaps: list[ClusterOverlap]