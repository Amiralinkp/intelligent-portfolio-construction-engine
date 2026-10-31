
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from intelligent_portfolio_construction_engine.features.behavioral_history_engine import (
    BehavioralHistoryEngine,
)


def test_build_history_runs_complete_pipeline(tmp_path: Path):
    settings = MagicMock()
    settings.BEHAVIORAL_HISTORY_START_DATE = "2020-01-01"
    settings.BEHAVIORAL_HISTORY_END_DATE = "2024-01-01"
    settings.BEHAVIORAL_SNAPSHOT_FREQUENCY = "W"

    provider = MagicMock()

    engine = BehavioralHistoryEngine(
        settings=settings,
        provider=provider,
        output_dir=tmp_path,
    )

    historical_series = MagicMock()

    asset_snapshots = pd.DataFrame(
        {
            "volatility": [0.2],
            "current_drawdown": [-0.1],
            "cagr": [0.12],
            "momentum_factor": [0.5],
            "beta": [1.1],
            "sharpe_ratio": [1.2],
            "sortino_ratio": [1.5],
            "symbol": ["AAPL"],
            "asset_class": ["equity"],
        },
        index=pd.to_datetime(["2023-12-31"]),
    )

    expected_history = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(["2023-12-31"]),
            "symbol": ["AAPL"],
            "asset_class": ["equity"],
            "volatility": [0.2],
            "current_drawdown": [-0.1],
            "cagr": [0.12],
            "momentum_factor": [0.5],
            "beta": [1.1],
            "sharpe_ratio": [1.2],
            "sortino_ratio": [1.5],
            "cluster_id": [0],
        }
    )

    with (
        patch(
            "intelligent_portfolio_construction_engine.features.behavioral_history_engine.get_benchmark"
        ) as mock_benchmark,
        patch.object(
            engine.context_engine,
            "build_feature_series",
            return_value=historical_series,
        ) as mock_features,
        patch(
            "intelligent_portfolio_construction_engine.features.behavioral_history_engine.build_behavioral_snapshots",
            return_value=asset_snapshots,
        ) as mock_snapshots,
        patch(
            "intelligent_portfolio_construction_engine.features.behavioral_history_engine.combine_behavioral_snapshots",
            return_value=expected_history.drop(columns="cluster_id"),
        ) as mock_combine,
        patch(
            "intelligent_portfolio_construction_engine.features.behavioral_history_engine.cluster_and_save_behavioral_history",
            return_value=expected_history,
        ) as mock_cluster,
    ):
        mock_benchmark.return_value.symbol = "^GSPC"

        result = engine.build_history(
            [{"symbol": "AAPL", "asset_class": "equity"}]
        )

    assert result.equals(expected_history)

    assert provider.get_data.call_count == 2
    mock_features.assert_called_once()
    mock_snapshots.assert_called_once()
    mock_combine.assert_called_once()
    mock_cluster.assert_called_once()

    assert (tmp_path / "behavioral_snapshots.csv").exists()
