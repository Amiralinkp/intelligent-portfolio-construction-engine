
import numpy as np
import pandas as pd
import pytest

from intelligent_portfolio_construction_engine.forecasting.dataset_analyzer import (
    DatasetAnalyzer,
)
from intelligent_portfolio_construction_engine.forecasting.dataset_builder import (
    FEATURE_COLUMNS,
    ForecastingDatasetBuilder,
)


def create_dataset(sample_count: int = 40) -> pd.DataFrame:
    records = []

    for index in range(sample_count):
        record = {
            "symbol": f"ASSET_{index % 5}",
            "snapshot_date": pd.Timestamp("2024-01-01")
            + pd.Timedelta(weeks=index),
            "target_date": pd.Timestamp("2024-01-01")
            + pd.Timedelta(weeks=index + 4),
            "model_version": "model_1",
            "target_model_version": "model_1",
            "cluster_id": index % 3,
            "target_cluster_id": index % 3,
        }

        record.update(
            {
                feature: float(index + 1)
                for feature in FEATURE_COLUMNS
            }
        )

        records.append(record)

    return pd.DataFrame(records)


def test_analyze_returns_expected_analysis_model() -> None:
    dataset = create_dataset()

    analyzer = DatasetAnalyzer()

    analysis = analyzer.analyze(dataset)

    assert isinstance(analysis.dataset_summary, dict)
    assert isinstance(analysis.quality_report, dict)
    assert isinstance(analysis.feature_statistics, pd.DataFrame)
    assert isinstance(analysis.target_distribution, pd.DataFrame)
    assert isinstance(analysis.warnings, list)
    assert isinstance(analysis.recommendations, list)


def test_analyze_calculates_statistics_for_all_features() -> None:
    dataset = create_dataset()

    analyzer = DatasetAnalyzer()

    analysis = analyzer.analyze(dataset)

    assert set(analysis.feature_statistics["feature"]) == set(
        FEATURE_COLUMNS
    )

    assert len(analysis.feature_statistics) == len(FEATURE_COLUMNS)

    assert (analysis.feature_statistics["count"] == 40).all()


def test_analyze_detects_small_dataset() -> None:
    dataset = create_dataset(sample_count=20)

    analyzer = DatasetAnalyzer()

    analysis = analyzer.analyze(dataset)

    assert any(
        "small" in warning.lower()
        or "sample" in warning.lower()
        for warning in analysis.warnings
    )


def test_analyze_detects_class_imbalance() -> None:
    dataset = create_dataset(sample_count=100)
    dataset["target_cluster_id"] = 0
    dataset.loc[90:, "target_cluster_id"] = 1

    analyzer = DatasetAnalyzer()

    analysis = analyzer.analyze(dataset)

    assert any(
        "imbalance" in warning.lower()
        or "minority" in warning.lower()
        or "dominant" in warning.lower()
        for warning in analysis.warnings
    )


def test_analyze_detects_non_finite_feature_values() -> None:
    dataset = create_dataset()
    dataset.loc[0, "volatility"] = np.inf

    analyzer = DatasetAnalyzer()
    analysis = analyzer.analyze(dataset)

    volatility_stats = analysis.feature_statistics.loc[
        analysis.feature_statistics["feature"] == "volatility"
    ].iloc[0]

    assert volatility_stats["non_finite_count"] == 1
    assert volatility_stats["count"] == 39


def test_analyze_detects_missing_feature_values() -> None:
    dataset = create_dataset()
    dataset.loc[0, "volatility"] = np.nan

    analyzer =DatasetAnalyzer()

    analysis = analyzer.analyze(dataset)

    volatility_stats = analysis.feature_statistics.loc[
        analysis.feature_statistics["feature"] == "volatility"
    ].iloc[0]

    assert volatility_stats["missing_count"] == 1
    assert volatility_stats["count"] == 39


def test_analyze_rejects_missing_required_columns() -> None:
    dataset = create_dataset()
    dataset = dataset.drop(columns=["target_cluster_id"])

    analyzer = DatasetAnalyzer()

    with pytest.raises(ValueError, match="Missing required columns"):
        analyzer.analyze(dataset)


def test_analyze_does_not_modify_original_dataset() -> None:
    dataset = create_dataset()
    original = dataset.copy(deep=True)

    analyzer = DatasetAnalyzer()

    analyzer.analyze(dataset)

    pd.testing.assert_frame_equal(dataset, original)
