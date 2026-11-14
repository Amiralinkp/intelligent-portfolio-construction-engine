
import pandas as pd

from intelligent_portfolio_construction_engine.forecasting.dataset_builder import (
    FEATURE_COLUMNS,
)
from intelligent_portfolio_construction_engine.forecasting.forecasting_dataset_pipeline import (
    ForecastingDatasetPipeline,
)


def create_behavioral_history(
    periods: int = 40,
) -> pd.DataFrame:
    dates = pd.date_range(
        start="2024-01-01",
        periods=periods,
        freq="7D",
    )

    records = []

    for asset_index, symbol in enumerate(
        ["BTC-USD", "ETH-USD", "GLD", "SPY", "QQQ"]
    ):
        for index, snapshot_date in enumerate(dates):
            record = {
                "symbol": symbol,
                "snapshot_date": snapshot_date,
                "model_version": f"model_{index // 10 + 1}",
                "cluster_id": (index + asset_index) % 3,
            }

            record.update(
                {
                    feature: float(index + asset_index + 1)
                    for feature in FEATURE_COLUMNS
                }
            )

            records.append(record)

    return pd.DataFrame(records)


def test_pipeline_generates_dataset_analysis_and_report(
    tmp_path,
) -> None:
    history = create_behavioral_history()

    pipeline = ForecastingDatasetPipeline(
        horizon_weeks=4,
        output_dir=tmp_path,
    )

    dataset, analysis, report_path = pipeline.run(history)

    assert not dataset.empty
    assert len(dataset) == 5 * (40 - 4)

    assert analysis.dataset_summary["sample_count"] == len(dataset)

    assert report_path.exists()
    assert (report_path / "report.html").exists()
    assert (report_path / "dataset_quality.json").exists()


def test_pipeline_preserves_source_behavioral_history(
    tmp_path,
) -> None:
    history = create_behavioral_history()
    original = history.copy(deep=True)

    pipeline = ForecastingDatasetPipeline(
        output_dir=tmp_path,
    )

    pipeline.run(history)

    pd.testing.assert_frame_equal(history, original)


def test_pipeline_creates_independent_reports_for_each_run(
    tmp_path,
) -> None:
    history = create_behavioral_history()

    pipeline = ForecastingDatasetPipeline(
        output_dir=tmp_path,
    )

    first_result = pipeline.run(history)
    second_result = pipeline.run(history)

    assert first_result[2] != second_result[2]
    assert first_result[2].exists()
    assert second_result[2].exists()


def test_pipeline_respects_custom_forecast_horizon(
    tmp_path,
) -> None:
    history = create_behavioral_history()

    pipeline = ForecastingDatasetPipeline(
        horizon_weeks=8,
        output_dir=tmp_path,
    )

    dataset, _, _ = pipeline.run(history)

    assert len(dataset) == 5 * (40 - 8)

    assert (
        dataset["target_date"] - dataset["snapshot_date"]
    ).eq(pd.Timedelta(weeks=8)).all()
