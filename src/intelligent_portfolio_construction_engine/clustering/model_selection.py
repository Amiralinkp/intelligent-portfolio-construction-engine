from dataclasses import dataclass
import pandas as pd
from intelligent_portfolio_construction_engine.clustering.evaluation import evaluate_clustering
from intelligent_portfolio_construction_engine.clustering.gmm import GMMClusterer
from intelligent_portfolio_construction_engine.clustering.hierarchical import HierarchicalClusterer
from intelligent_portfolio_construction_engine.clustering.kmeans import KMeansClusterer
from intelligent_portfolio_construction_engine.models.clustering_result import ClusteringResult


@dataclass(frozen=True)
class ModelSelectionResult:

    model_name: str
    n_clusters: int
    clustering_result: ClusteringResult
    metrics: dict[str, float]
    composite_score: float


class ClusteringModelSelector:

    def __init__(self, min_k: int, max_k: int):
        self.min_k = min_k
        self.max_k = max_k

    def select(self, data):

        candidates = []

        for n_clusters in range(self.min_k, self.max_k + 1):
            models = [
                KMeansClusterer(n_clusters=n_clusters),
                HierarchicalClusterer(n_clusters=n_clusters),
                GMMClusterer(n_clusters=n_clusters)]

            for model in models:
                result = model.fit_predict(data)

                metrics = evaluate_clustering(data=data, result=result)
                composite_score = self._calculate_composite_score(metrics)

                candidates.append(
                    ModelSelectionResult(
                        model_name=result.model_name,
                        n_clusters=n_clusters,
                        clustering_result=result,
                        metrics=metrics,
                        composite_score=composite_score))

        best_result = max(candidates, key=lambda candidate: candidate.composite_score)

        return best_result, candidates

    @staticmethod
    def _calculate_composite_score(metrics):

        silhouette = metrics["silhouette"]
        calinski_harabasz = metrics["calinski_harabasz"]
        davies_bouldin = metrics["davies_bouldin"]

        normalized_calinski = (calinski_harabasz / (1 + calinski_harabasz))
        normalized_davies_bouldin = (1 / (1 + davies_bouldin))

        return (silhouette + normalized_calinski + normalized_davies_bouldin) / 3