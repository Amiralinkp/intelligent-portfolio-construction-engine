from pathlib import Path
from intelligent_portfolio_construction_engine.config.setting import Settings
import numpy as np
import pandas as pd

from intelligent_portfolio_construction_engine.clustering.clustering_runner import (
    run_clustering_models,
)


class TestSettings:
    CLUSTERING_MIN_K = 2
    CLUSTERING_MAX_K = 4


def create_clustering_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)

    cluster_1 = rng.normal(
        loc=0.0,
        scale=0.4,
        size=(30, 7),
    )

    cluster_2 = rng.normal(
        loc=3.0,
        scale=0.4,
        size=(30, 7),
    )

    cluster_3 = rng.normal(
        loc=6.0,
        scale=0.4,
        size=(30, 7),
    )

    data = np.vstack(
        [
            cluster_1,
            cluster_2,
            cluster_3,
        ]
    )

    return pd.DataFrame(
        data,
        columns=[
            "volatility",
            "current_drawdown",
            "cagr",
            "momentum_factor",
            "beta",
            "sharpe_ratio",
            "sortino_ratio",
        ],
    )


def test_clustering_model_selection(
    tmp_path: Path,
):
    data = create_clustering_data()

    output_dir = Path(
        "src/intelligent_portfolio_construction_engine/tests/visualizations"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_result, candidates = run_clustering_models(
        data=data,
        settings=TestSettings(),
        output_dir=output_dir,
    )

    min_k = TestSettings.CLUSTERING_MIN_K
    max_k = TestSettings.CLUSTERING_MAX_K

    expected_candidate_count = (
        (max_k - min_k + 1) * 3
    )

    assert len(candidates) == expected_candidate_count

    expected_models = {
        "K-Means",
        "Hierarchical",
        "GMM",
    }

    actual_models = {
        candidate.model_name
        for candidate in candidates
    }

    assert actual_models == expected_models

    actual_k_values = {
        candidate.n_clusters
        for candidate in candidates
    }

    expected_k_values = set(
        range(
            min_k,
            max_k + 1,
        )
    )

    assert actual_k_values == expected_k_values

    for candidate in candidates:
        assert candidate.metrics.keys() == {
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
        }

        assert np.isfinite(
            candidate.composite_score
        )

        assert len(
            candidate.clustering_result.labels
        ) == len(data)

    assert best_result in candidates

    expected_visualizations = (
        (max_k - min_k + 1) * 3
    )

    visualization_files = list(
    output_dir.glob("*.png")
)

    assert len(visualization_files) == (
        expected_visualizations
    )