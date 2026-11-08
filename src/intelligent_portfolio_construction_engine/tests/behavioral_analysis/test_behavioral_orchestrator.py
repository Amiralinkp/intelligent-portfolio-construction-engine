
import pandas as pd

from intelligent_portfolio_construction_engine.behavioral_analysis import (
    analyze_behavioral_history,
)
from intelligent_portfolio_construction_engine.models import (
    BehavioralAnalysisResult,
    StateTransitionResult,
)


def test_behavioral_orchestrator_returns_complete_result():
    history = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(
                [
                    "2024-01-07",
                    "2024-01-07",
                    "2024-01-14",
                    "2024-01-14",
                ]
            ),
            "symbol": ["BTC", "ETH", "BTC", "ETH"],
            "model_version": ["v1", "v1", "v1", "v1"],
            "cluster_id": [0, 1, 1, 1],
            "volatility": [0.2, 0.3, 0.25, 0.28],
            "current_drawdown": [-0.1, -0.2, -0.15, -0.18],
            "cagr": [0.15, 0.12, 0.18, 0.14],
            "momentum_factor": [0.1, 0.2, 0.15, 0.18],
            "beta": [1.2, 0.8, 1.1, 0.85],
            "sharpe_ratio": [1.0, 0.8, 1.2, 0.9],
            "sortino_ratio": [1.3, 1.0, 1.5, 1.1],
        }
    )

    original_history = history.copy(deep=True)

    result = analyze_behavioral_history(history)

    assert isinstance(result, BehavioralAnalysisResult)
    assert isinstance(result.state_transition, StateTransitionResult)

    assert not result.cluster_characteristics.empty
    assert not result.market_behavior_distribution.empty
    assert not result.state_transition.asset_state_paths.empty

    pd.testing.assert_frame_equal(history, original_history)


def test_behavioral_orchestrator_preserves_model_versions():
    history = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(
                [
                    "2024-01-07",
                    "2024-01-14",
                ]
            ),
            "symbol": ["BTC", "BTC"],
            "model_version": ["v1", "v2"],
            "cluster_id": [0, 0],
            "volatility": [0.2, 0.25],
            "current_drawdown": [-0.1, -0.12],
            "cagr": [0.15, 0.18],
            "momentum_factor": [0.1, 0.12],
            "beta": [1.2, 1.1],
            "sharpe_ratio": [1.0, 1.2],
            "sortino_ratio": [1.3, 1.5],
        }
    )

    result = analyze_behavioral_history(history)

    versions = result.cluster_characteristics["model_version"]

    assert set(versions) == {"v1", "v2"}
