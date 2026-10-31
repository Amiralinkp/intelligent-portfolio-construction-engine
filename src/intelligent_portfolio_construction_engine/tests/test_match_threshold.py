import numpy as np
import pandas as pd
import pytest

from intelligent_portfolio_construction_engine.clustering.cluster_profiling import (
    FEATURE_COLUMNS,
)
from intelligent_portfolio_construction_engine.clustering.match_threshold import (
    BootstrapThresholdEstimator,
)


@pytest.fixture
def estimator():
    return BootstrapThresholdEstimator(
        n_bootstrap=500,
        confidence_level=0.95,
        random_state=42,
    )


@pytest.fixture
def reference_scale():
    return pd.Series(
        {feature: 1.0 for feature in FEATURE_COLUMNS}
    )


@pytest.fixture
def cluster_features():
    rng = np.random.default_rng(42)

    return pd.DataFrame(
        rng.normal(
            loc=0.0,
            scale=1.0,
            size=(100, len(FEATURE_COLUMNS)),
        ),
        columns=FEATURE_COLUMNS,
    )


def test_estimate_returns_positive_threshold(
    estimator,
    cluster_features,
    reference_scale,
):
    threshold = estimator.estimate(
        cluster_features,
        reference_scale,
    )

    assert threshold > 0
    assert np.isfinite(threshold)


def test_estimate_is_reproducible(
    cluster_features,
    reference_scale,
):
    first = BootstrapThresholdEstimator(
        n_bootstrap=500,
        random_state=42,
    )
    second = BootstrapThresholdEstimator(
        n_bootstrap=500,
        random_state=42,
    )

    assert first.estimate(
        cluster_features,
        reference_scale,
    ) == second.estimate(
        cluster_features,
        reference_scale,
    )


def test_missing_feature_raises_error(
    estimator,
    cluster_features,
    reference_scale,
):
    invalid_features = cluster_features.drop(
        columns=[FEATURE_COLUMNS[0]]
    )

    with pytest.raises(ValueError, match="Missing cluster features"):
        estimator.estimate(
            invalid_features,
            reference_scale,
        )


def test_invalid_reference_scale_raises_error(
    estimator,
    cluster_features,
):
    invalid_scale = pd.Series(
        {feature: 1.0 for feature in FEATURE_COLUMNS}
    )
    invalid_scale[FEATURE_COLUMNS[0]] = 0.0

    with pytest.raises(ValueError, match="finite and positive"):
        estimator.estimate(
            cluster_features,
            invalid_scale,
        )


def test_single_observation_raises_error(
    estimator,
    reference_scale,
):
    features = pd.DataFrame(
        [[1.0] * len(FEATURE_COLUMNS)],
        columns=FEATURE_COLUMNS,
    )

    with pytest.raises(ValueError, match="At least two observations"):
        estimator.estimate(
            features,
            reference_scale,
        )


def test_non_finite_features_raise_error(
    estimator,
    cluster_features,
    reference_scale,
):
    invalid_features = cluster_features.copy()
    invalid_features.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match="must be finite"):
        estimator.estimate(
            invalid_features,
            reference_scale,
        )