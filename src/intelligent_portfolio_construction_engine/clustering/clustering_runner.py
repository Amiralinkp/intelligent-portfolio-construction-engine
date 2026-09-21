from pathlib import Path
import pandas as pd
from intelligent_portfolio_construction_engine.clustering.model_selection import ClusteringModelSelector
from intelligent_portfolio_construction_engine.clustering.visualization import save_cluster_visualization


def run_clustering_models(data, settings, output_dir: Path):

    selector = ClusteringModelSelector(min_k=settings.CLUSTERING_MIN_K, max_k=settings.CLUSTERING_MAX_K)
    best_result, candidates = selector.select(data)

    for candidate in candidates:
        save_cluster_visualization(data=data, result=candidate.clustering_result,
                                    output_dir=output_dir, n_clusters=candidate.n_clusters)

    return best_result, candidates