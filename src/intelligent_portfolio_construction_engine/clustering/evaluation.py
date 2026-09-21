import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from intelligent_portfolio_construction_engine.models.clustering_result import ClusteringResult


def evaluate_clustering(data, result: ClusteringResult):

    labels = result.labels
    return {
        "silhouette": silhouette_score(data, labels),
        "calinski_harabasz": calinski_harabasz_score(data, labels),
        "davies_bouldin": davies_bouldin_score(data, labels)}