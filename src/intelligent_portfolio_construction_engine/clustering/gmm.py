import pandas as pd
from sklearn.mixture import GaussianMixture
from intelligent_portfolio_construction_engine.models.clustering_result import ClusteringResult

class GMMClusterer:
    def __init__(self, n_clusters: int, random_state: int = 42):

        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit_predict(self, data) :

        model = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)

        labels = model.fit_predict(data)
        label_series = pd.Series(labels, index=data.index, name="cluster")

        return ClusteringResult(
            labels=label_series,
            model_name="GMM")