import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from intelligent_portfolio_construction_engine.models.clustering_result import (
    ClusteringResult,
)


class HierarchicalClusterer:

    def __init__(self, n_clusters: int, linkage: str = "ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.model = None
        self.cluster_centroids = None
        self.feature_columns = None

    def fit_predict(self, data: pd.DataFrame) -> ClusteringResult:
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
        )

        labels = self.model.fit_predict(data)

        self.feature_columns = data.columns

        training_data = data.copy()
        training_data["cluster"] = labels

        self.cluster_centroids = (
            training_data.groupby("cluster")[self.feature_columns]
            .mean()
        )

        label_series = pd.Series(
            labels,
            index=data.index,
            name="cluster",
        )

        return ClusteringResult(
            labels=label_series,
            model_name="Hierarchical",
            estimator=self,
        )

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if self.cluster_centroids is None:
            raise RuntimeError("The clustering model has not been fitted.")

        if not data.columns.equals(self.feature_columns):
            raise ValueError("Input features do not match the training features.")

        values = data.to_numpy(dtype=float)
        centroids = self.cluster_centroids.to_numpy(dtype=float)

        distances = np.linalg.norm(
            values[:, np.newaxis, :] - centroids[np.newaxis, :, :],
            axis=2,
        )

        nearest_positions = distances.argmin(axis=1)
        labels = self.cluster_centroids.index.to_numpy()[nearest_positions]

        return pd.Series(
            labels,
            index=data.index,
            name="cluster",
        )