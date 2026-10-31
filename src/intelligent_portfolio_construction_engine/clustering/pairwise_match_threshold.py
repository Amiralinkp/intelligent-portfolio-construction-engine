import numpy as np
import pandas as pd
from intelligent_portfolio_construction_engine.models.match_threshold import (
    MatchThreshold,
)
from intelligent_portfolio_construction_engine.clustering.cluster_profiling import (
    FEATURE_COLUMNS,
)


class PairwiseMatchThresholdEstimator:
    def __init__(
        self,
        n_bootstrap: int = 1000,
        significance_level: float = 0.05,
        random_state: int = 42,
    ) -> None:
        if n_bootstrap < 100:
            raise ValueError("n_bootstrap must be at least 100.")

        if not 0 < significance_level < 1:
            raise ValueError("significance_level must be between 0 and 1.")

        self.n_bootstrap = n_bootstrap
        self.significance_level = significance_level
        self.random_state = random_state

    def estimate(
    self,
    previous_features: pd.DataFrame,
    current_features: pd.DataFrame,
    reference_scale: pd.Series,
) :
        self._validate_inputs(
            previous_features,
            current_features,
            reference_scale,
        )

        previous = previous_features[FEATURE_COLUMNS].to_numpy(
            dtype=float
        )
        current = current_features[FEATURE_COLUMNS].to_numpy(
            dtype=float
        )
        scale = reference_scale[FEATURE_COLUMNS].to_numpy(
            dtype=float
        )

        pooled = np.vstack([previous, current])

        previous_size = len(previous)
        current_size = len(current)

        rng = np.random.default_rng(self.random_state)
        null_distances = np.empty(self.n_bootstrap)

        for index in range(self.n_bootstrap):
            previous_sample = pooled[
                rng.integers(0, len(pooled), size=previous_size)
            ]
            current_sample = pooled[
                rng.integers(0, len(pooled), size=current_size)
            ]

            previous_centroid = previous_sample.mean(axis=0)
            current_centroid = current_sample.mean(axis=0)

            null_distances[index] = np.linalg.norm(
                (previous_centroid - current_centroid) / scale
            )

        return MatchThreshold(
            threshold_95=float(np.quantile(null_distances, 0.95)),
            threshold_99=float(np.quantile(null_distances, 0.99)),
        )

    @staticmethod
    def _validate_inputs(
        previous_features: pd.DataFrame,
        current_features: pd.DataFrame,
        reference_scale: pd.Series,
    ) -> None:
        for name, features in (
            ("previous_features", previous_features),
            ("current_features", current_features),
        ):
            missing = set(FEATURE_COLUMNS) - set(features.columns)

            if missing:
                raise ValueError(
                    f"{name} is missing features: {sorted(missing)}"
                )

            if len(features) < 2:
                raise ValueError(
                    f"{name} must contain at least two observations."
                )

            values = features[FEATURE_COLUMNS].to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"{name} must contain only finite values."
                )

        missing_scale = set(FEATURE_COLUMNS) - set(reference_scale.index)

        if missing_scale:
            raise ValueError(
                f"Missing reference scale features: {sorted(missing_scale)}"
            )

        scale = reference_scale[FEATURE_COLUMNS].to_numpy(dtype=float)

        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError(
                "Reference scale values must be finite and positive."
            )