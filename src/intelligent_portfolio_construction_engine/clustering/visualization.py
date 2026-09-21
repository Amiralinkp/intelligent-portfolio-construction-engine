from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from intelligent_portfolio_construction_engine.models.clustering_result import ClusteringResult

def save_cluster_visualization(data, result: ClusteringResult, output_dir: Path, n_clusters: int) -> Path:
    pca = PCA(n_components=2)
    projected_data = pca.fit_transform(data)

    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = result.model_name.lower().replace(" ", "_")

    output_path = (output_dir / f"{model_name}_k{n_clusters}_clusters.png")

    plt.figure(figsize=(10, 7))

    plt.scatter(
        projected_data[:, 0],
        projected_data[:, 1],
        c=result.labels.to_numpy(),
        alpha=0.7)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"{result.model_name} Clusters (k={n_clusters})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path