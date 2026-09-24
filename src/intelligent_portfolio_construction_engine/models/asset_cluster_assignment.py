from dataclasses import dataclass



@dataclass(frozen=True)
class AssetClusterAssignment:

    asset: str
    cluster_id: int