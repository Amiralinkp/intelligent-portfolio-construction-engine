from dataclasses import dataclass

@dataclass(frozen=True)
class ClusterProfile:
    cluster_id: int
    asset_count: int
    centroid: dict[str, float]
    median_profile: dict[str, float]