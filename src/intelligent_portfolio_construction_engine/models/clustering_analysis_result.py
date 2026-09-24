from dataclasses import dataclass
from intelligent_portfolio_construction_engine.models.asset_cluster_assignment import AssetClusterAssignment
from intelligent_portfolio_construction_engine.models.cluster_interpretation import ClusterInterpretation
from intelligent_portfolio_construction_engine.models.cluster_profile import ClusterProfile
from intelligent_portfolio_construction_engine.clustering.model_selection import ModelSelectionResult


@dataclass(frozen=True)


class ClusteringAnalysisResult:

    model_selection: ModelSelectionResult
    cluster_profiles: list[ClusterProfile]
    candidates: list
    cluster_interpretations: list[ClusterInterpretation]
    asset_cluster_assignments: list[AssetClusterAssignment]