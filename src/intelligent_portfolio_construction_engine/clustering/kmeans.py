import pandas as pd
from sklearn.cluster import KMeans

from intelligent_portfolio_construction_engine.models.clustering_result import (
    ClusteringResult,
)


class KMeansClusterer:

    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None

    def fit_predict(self, data: pd.DataFrame) -> ClusteringResult:
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )

        labels = self.model.fit_predict(data)

        label_series = pd.Series(
            labels,
            index=data.index,
            name="cluster",
        )

        return ClusteringResult(
            labels=label_series,
            model_name="K-Means",
            estimator=self.model,
        )

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("The clustering model has not been fitted.")

        labels = self.model.predict(data)

        return pd.Series(
            labels,
            index=data.index,
            name="cluster",
        )