import pandas as pd
import pytest

from intelligent_portfolio_construction_engine.clustering.cluster_interpreter import (
    interpret_cluster_profiles,
)
from intelligent_portfolio_construction_engine.clustering.cluster_profiling import (
    FEATURE_COLUMNS,
    build_cluster_profiles,
)


@pytest.fixture
def sample_data():
    features = pd.DataFrame(
        [
            [0.10, -0.05, 0.12, 0.08, 1.1, 1.2, 1.4],
            [0.20, -0.10, 0.08, 0.04, 0.9, 0.8, 1.0],
            [0.30, -0.15, 0.04, 0.02, 0.7, 0.5, 0.6],
            [0.40, -0.20, 0.02, 0.01, 0.5, 0.3, 0.4],
        ],
        columns=FEATURE_COLUMNS,
    )

    labels = pd.Series([0, 0, 1, 1])

    return features, labels


def test_build_cluster_profiles(sample_data):
    features, labels = sample_data

    profiles = build_cluster_profiles(features, labels)

    assert len(profiles) == 2

    first_profile = profiles[0]

    assert first_profile.cluster_id == 0
    assert first_profile.asset_count == 2

    assert first_profile.centroid["volatility"] == pytest.approx(0.15)
    assert first_profile.median_profile["volatility"] == pytest.approx(0.15)

    assert first_profile.centroid["cagr"] == pytest.approx(0.10)
    assert first_profile.median_profile["cagr"] == pytest.approx(0.10)

    assert set(first_profile.centroid) == set(FEATURE_COLUMNS)
    assert set(first_profile.median_profile) == set(FEATURE_COLUMNS)


def test_interpret_cluster_profiles(sample_data):
    features, labels = sample_data

    profiles = build_cluster_profiles(features, labels)
    interpretations = interpret_cluster_profiles(profiles)

    assert len(interpretations) == 2

    first = interpretations[0]
    second = interpretations[1]

    assert first.cluster_id == 0
    assert second.cluster_id == 1

    assert first.volatility_level == "low"
    assert second.volatility_level == "high"

    assert first.drawdown_level == "high"
    assert second.drawdown_level == "low"

    assert first.return_profile == "high"
    assert second.return_profile == "low"

    assert first.momentum_profile == "high"
    assert second.momentum_profile == "low"


def test_single_cluster_interpretation(sample_data):
    features, _ = sample_data

    labels = pd.Series([0, 0, 0, 0])

    profiles = build_cluster_profiles(features, labels)
    interpretations = interpret_cluster_profiles(profiles)

    assert len(interpretations) == 1

    interpretation = interpretations[0]

    assert interpretation.volatility_level == "medium"
    assert interpretation.drawdown_level == "medium"
    assert interpretation.beta_level == "medium"
    assert interpretation.return_profile == "medium"
    assert interpretation.momentum_profile == "medium"
    assert interpretation.sharpe_level == "medium"
    assert interpretation.sortino_level == "medium"