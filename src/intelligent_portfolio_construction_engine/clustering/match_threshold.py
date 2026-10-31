import numpy as np
import pandas as pd

from intelligent_portfolio_construction_engine.clustering.cluster_profiling import (
    FEATURE_COLUMNS,
)


class BootstrapThresholdEstimator:
    def __init__(
        self,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: int = 42,
    ) -> None:
        if n_bootstrap < 100:
            raise ValueError("n_bootstrap must be at least 100.")

        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1.")

        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state

    def estimate(
        self,
        cluster_features: pd.DataFrame,
        reference_scale: pd.Series,
    ) -> float:
        self._validate_inputs(cluster_features, reference_scale)

        values = cluster_features[FEATURE_COLUMNS].to_numpy(dtype=float)
        scale = reference_scale[FEATURE_COLUMNS].to_numpy(dtype=float)

        rng = np.random.default_rng(self.random_state)
        distances = []

        for _ in range(self.n_bootstrap):
            first_sample = values[
                rng.integers(0, len(values), size=len(values))
            ]
            second_sample = values[
                rng.integers(0, len(values), size=len(values))
            ]

            first_centroid = first_sample.mean(axis=0)
            second_centroid = second_sample.mean(axis=0)

            distance = np.linalg.norm(
                (first_centroid - second_centroid) / scale
            )
            distances.append(distance)

        return float(
            np.quantile(distances, self.confidence_level)
        )

    @staticmethod
    def _validate_inputs(
        cluster_features: pd.DataFrame,
        reference_scale: pd.Series,
    ) -> None:
        missing_features = set(FEATURE_COLUMNS) - set(
            cluster_features.columns
        )
        if missing_features:
            raise ValueError(
                f"Missing cluster features: {sorted(missing_features)}"
            )

        missing_scale = set(FEATURE_COLUMNS) - set(reference_scale.index)
        if missing_scale:
            raise ValueError(
                f"Missing reference scale features: {sorted(missing_scale)}"
            )

        if len(cluster_features) < 2:
            raise ValueError(
                "At least two observations are required."
            )

        values = cluster_features[FEATURE_COLUMNS].to_numpy(dtype=float)
        scale = reference_scale[FEATURE_COLUMNS].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError("Cluster features must be finite.")

        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError(
                "Reference scale values must be finite and positive."
            )