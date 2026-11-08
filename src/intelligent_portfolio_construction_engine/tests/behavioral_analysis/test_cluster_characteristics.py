import numpy as np
import pandas as pd
import pytest

from intelligent_portfolio_construction_engine.behavioral_analysis import (
    analyze_cluster_characteristics,
)


@pytest.fixture
def behavioral_history():
    return pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(
                [
                    "2024-01-07",
                    "2024-01-07",
                    "2024-01-07",
                    "2024-01-07",
                ]
            ),
            "symbol": ["BTC", "ETH", "GOLD", "SP500"],
            "model_version": ["v1", "v1", "v1", "v1"],
            "cluster_id": [0, 0, 1, 1],
            "volatility": [0.2, 0.4, 0.1, 0.3],
            "current_drawdown": [-0.1, -0.2, -0.05, -0.15],
            "cagr": [0.15, 0.25, 0.08, 0.12],
            "momentum_factor": [0.1, 0.2, 0.05, 0.15],
            "beta": [1.2, 1.4, 0.3, 1.0],
            "sharpe_ratio": [1.0, 1.5, 0.8, 1.2],
            "sortino_ratio": [1.3, 1.8, 1.0, 1.5],
        }
    )


def test_cluster_characteristics_calculates_statistics(
    behavioral_history,
):
    result = analyze_cluster_characteristics(
        behavioral_history
    )

    cluster_zero = result.loc[
        result["cluster_id"] == 0
    ].iloc[0]

    assert cluster_zero["asset_count"] == 2
    assert cluster_zero["volatility_mean"] == pytest.approx(0.3)
    assert cluster_zero["volatility_median"] == pytest.approx(0.3)


def test_cluster_characteristics_preserves_group_identity(
    behavioral_history,
):
    result = analyze_cluster_characteristics(
        behavioral_history
    )

    assert len(result) == 2
    assert set(result["cluster_id"]) == {0, 1}
    assert set(result["model_version"]) == {"v1"}


def test_cluster_characteristics_rejects_empty_input(
    behavioral_history,
):
    with pytest.raises(ValueError):
        analyze_cluster_characteristics(
            behavioral_history.iloc[0:0]
        )


def test_cluster_characteristics_rejects_duplicate_snapshots(
    behavioral_history,
):
    duplicated = pd.concat(
        [behavioral_history, behavioral_history.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError):
        analyze_cluster_characteristics(duplicated)


def test_cluster_characteristics_rejects_non_finite_features(
    behavioral_history,
):
    invalid = behavioral_history.copy()
    invalid.loc[0, "volatility"] = np.inf

    with pytest.raises(ValueError):
        analyze_cluster_characteristics(invalid)