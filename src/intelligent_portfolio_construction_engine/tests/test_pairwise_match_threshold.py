
import numpy as np
import pandas as pd
import pytest

from intelligent_portfolio_construction_engine.clustering.cluster_profiling import (
    FEATURE_COLUMNS,
)
from intelligent_portfolio_construction_engine.clustering.pairwise_match_threshold import (
    PairwiseMatchThresholdEstimator,
)


@pytest.fixture
def estimator():
    return PairwiseMatchThresholdEstimator(
        n_bootstrap=500,
        significance_level=0.05,
        random_state=42,
    )


@pytest.fixture
def reference_scale():
    return pd.Series(
        {feature: 1.0 for feature in FEATURE_COLUMNS}
    )


@pytest.fixture
def feature_samples():
    rng = np.random.default_rng(42)

    previous = pd.DataFrame(
        rng.normal(0, 1, size=(100, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )

    current = pd.DataFrame(
        rng.normal(0, 1, size=(100, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )

    return previous, current


def test_estimate_returns_valid_thresholds(
    estimator,
    feature_samples,
    reference_scale,
):
    previous, current = feature_samples

    thresholds = estimator.estimate(
        previous,
        current,
        reference_scale,
    )

    assert thresholds.threshold_95 > 0
    assert thresholds.threshold_99 > thresholds.threshold_95
    assert np.isfinite(thresholds.threshold_95)
    assert np.isfinite(thresholds.threshold_99)


def test_estimate_is_reproducible(
    feature_samples,
    reference_scale,
):
    previous, current = feature_samples

    first = PairwiseMatchThresholdEstimator(
        n_bootstrap=500,
        random_state=42,
    )

    second = PairwiseMatchThresholdEstimator(
        n_bootstrap=500,
        random_state=42,
    )

    first_threshold = first.estimate(
        previous,
        current,
        reference_scale,
    )

    second_threshold = second.estimate(
        previous,
        current,
        reference_scale,
    )

    assert first_threshold == second_threshold


def test_missing_feature_raises_error(
    estimator,
    feature_samples,
    reference_scale,
):
    previous, current = feature_samples

    invalid_previous = previous.drop(
        columns=[FEATURE_COLUMNS[0]]
    )

    with pytest.raises(ValueError, match="is missing features"):
        estimator.estimate(
            invalid_previous,
            current,
            reference_scale,
        )


def test_invalid_reference_scale_raises_error(
    estimator,
    feature_samples,
):
    previous, current = feature_samples

    invalid_scale = pd.Series(
        {feature: 1.0 for feature in FEATURE_COLUMNS}
    )

    invalid_scale[FEATURE_COLUMNS[0]] = 0.0

    with pytest.raises(ValueError, match="finite and positive"):
        estimator.estimate(
            previous,
            current,
            invalid_scale,
        )


def test_single_observation_raises_error(
    estimator,
    feature_samples,
    reference_scale,
):
    previous, current = feature_samples

    invalid_previous = previous.iloc[:1]

    with pytest.raises(ValueError, match="at least two observations"):
        estimator.estimate(
            invalid_previous,
            current,
            reference_scale,
        )


def test_non_finite_features_raise_error(
    estimator,
    feature_samples,
    reference_scale,
):
    previous, current = feature_samples

    invalid_current = current.copy()
    invalid_current.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite values"):
        estimator.estimate(
            previous,
            invalid_current,
            reference_scale,
        )


def test_larger_samples_produce_smaller_threshold(
    reference_scale,
):
    rng = np.random.default_rng(123)

    small_previous = pd.DataFrame(
        rng.normal(0, 1, size=(30, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )

    small_current = pd.DataFrame(
        rng.normal(0, 1, size=(30, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )

    large_previous = pd.DataFrame(
        rng.normal(0, 1, size=(300, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )

    large_current = pd.DataFrame(
        rng.normal(0, 1, size=(300, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )

    estimator = PairwiseMatchThresholdEstimator(
        n_bootstrap=1000,
        random_state=42,
    )

    small_threshold = estimator.estimate(
        small_previous,
        small_current,
        reference_scale,
    )

    large_threshold = estimator.estimate(
        large_previous,
        large_current,
        reference_scale,
    )

    assert (
        large_threshold.threshold_95
        < small_threshold.threshold_95
    )
