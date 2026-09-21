import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from intelligent_portfolio_construction_engine.models.clustering_result import ClusteringResult



class HierarchicalClusterer:
    def __init__(self, n_clusters: int, linkage: str = "ward"):

        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit_predict(self, data):
        
        model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)

        labels = model.fit_predict(data)
        label_series = pd.Series(labels, index=data.index, name="cluster")

        return ClusteringResult(
            labels=label_series,
            model_name="Hierarchical")