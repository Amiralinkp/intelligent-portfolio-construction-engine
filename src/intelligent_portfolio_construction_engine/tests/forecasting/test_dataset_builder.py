
import pandas as pd
import pytest

from intelligent_portfolio_construction_engine.forecasting.dataset_builder import (
    FEATURE_COLUMNS,
    ForecastingDatasetBuilder,
)


def create_behavioral_history(
    periods: int = 10,
    missing_dates: list[int] | None = None,
) -> pd.DataFrame:
    dates = pd.date_range(
        start="2024-01-01",
        periods=periods,
        freq="7D",
    )

    if missing_dates:
        dates = dates.delete(missing_dates)

    records = []

    for index, snapshot_date in enumerate(dates):
        record = {
            "symbol": "AAPL",
            "snapshot_date": snapshot_date,
            "model_version": f"model_{index // 5 + 1}",
            "cluster_id": index % 3,
        }

        record.update(
            {
                feature: float(index + 1)
                for feature in FEATURE_COLUMNS
            }
        )

        records.append(record)

    return pd.DataFrame(records)


def test_build_creates_samples_using_exact_forecast_horizon() -> None:
    history = create_behavioral_history(periods=10)

    builder = ForecastingDatasetBuilder(
        horizon_weeks=4,
    )

    dataset = builder.build(history)

    assert len(dataset) == 6

    first_sample = dataset.iloc[0]

    assert first_sample["snapshot_date"] == pd.Timestamp("2024-01-01")
    assert first_sample["target_date"] == pd.Timestamp("2024-01-29")
    assert first_sample["target_cluster_id"] == 1


def test_build_skips_samples_with_missing_intermediate_snapshots() -> None:
    history = create_behavioral_history(
        periods=10,
        missing_dates=[2],
    )

    builder = ForecastingDatasetBuilder(
        horizon_weeks=4,
    )

    dataset = builder.build(history)

    assert pd.Timestamp("2024-01-01") not in dataset["snapshot_date"].values


def test_build_preserves_source_and_target_model_versions() -> None:
    history = create_behavioral_history(periods=10)

    builder = ForecastingDatasetBuilder(
        horizon_weeks=4,
    )

    dataset = builder.build(history)

    sample = dataset.loc[
        dataset["snapshot_date"] == pd.Timestamp("2024-01-22")
    ].iloc[0]

    assert sample["model_version"] == "model_1"
    assert sample["target_model_version"] == "model_2"


def test_build_returns_empty_dataset_when_history_is_too_short() -> None:
    history = create_behavioral_history(periods=4)

    builder = ForecastingDatasetBuilder(
        horizon_weeks=4,
    )

    dataset = builder.build(history)

    assert dataset.empty
    assert list(dataset.columns) == [
        "symbol",
        "snapshot_date",
        "target_date",
        "model_version",
        "target_model_version",
        "cluster_id",
        *FEATURE_COLUMNS,
        "target_cluster_id",
    ]


def test_build_rejects_duplicate_asset_snapshots() -> None:
    history = create_behavioral_history(periods=6)
    history = pd.concat(
        [history, history.iloc[[0]]],
        ignore_index=True,
    )

    builder = ForecastingDatasetBuilder()

    with pytest.raises(
        ValueError,
        match="Duplicate asset snapshots",
    ):
        builder.build(history)


def test_build_rejects_non_finite_features() -> None:
    history = create_behavioral_history(periods=6)
    history.loc[0, "volatility"] = float("inf")

    builder = ForecastingDatasetBuilder()

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        builder.build(history)


def test_build_does_not_modify_original_history() -> None:
    history = create_behavioral_history(periods=10)
    original = history.copy(deep=True)

    builder = ForecastingDatasetBuilder()

    builder.build(history)

    pd.testing.assert_frame_equal(history, original)
