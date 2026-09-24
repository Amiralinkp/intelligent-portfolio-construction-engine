from pathlib import Path
import pandas as pd
from intelligent_portfolio_construction_engine.clustering.model_selection import ClusteringModelSelector
from intelligent_portfolio_construction_engine.clustering.visualization import save_cluster_visualization
from intelligent_portfolio_construction_engine.clustering.cluster_profiling import build_cluster_profiles
from intelligent_portfolio_construction_engine.clustering.cluster_interpreter import interpret_cluster_profiles
from intelligent_portfolio_construction_engine.clustering.asset_cluster_assignment import build_asset_cluster_assignments
from intelligent_portfolio_construction_engine.models.clustering_analysis_result import ClusteringAnalysisResult


def run_clustering_models(data, settings, output_dir: Path):

    selector = ClusteringModelSelector(min_k=settings.CLUSTERING_MIN_K, max_k=settings.CLUSTERING_MAX_K)
    best_result, candidates = selector.select(data)
    cluster_profiles = build_cluster_profiles(features=data, labels=best_result.clustering_result.labels)
    cluster_interpretations = interpret_cluster_profiles(cluster_profiles)
    asset_cluster_assignments = build_asset_cluster_assignments(assets=data.index, labels=best_result.clustering_result.labels)
    for candidate in candidates:
        save_cluster_visualization(data=data, result=candidate.clustering_result,
                                    output_dir=output_dir, n_clusters=candidate.n_clusters)

    return ClusteringAnalysisResult(
        model_selection=best_result,
        cluster_profiles=cluster_profiles,
        candidates=candidates,
        cluster_interpretations=cluster_interpretations,
        asset_cluster_assignments=asset_cluster_assignments)  
    