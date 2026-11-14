
from datetime import datetime

import pandas as pd
import pytest

from intelligent_portfolio_construction_engine.forecasting.evaluation.evaluation_pipeline import (
    EvaluationPipeline,
)
from intelligent_portfolio_construction_engine.models.evaluation_run_metadata import (
    EvaluationRunMetadata,
)


class DummyForecastingModel:
    def __init__(
        self,
        predictions: list[int] | pd.Series | None = None,
    ) -> None:
        self.predictions = predictions
        self.is_fitted = False
        self.training_symbols: list[str] = []

    def fit(
        self,
        training_dataset: pd.DataFrame,
    ) -> "DummyForecastingModel":
        self.is_fitted = True
        self.training_symbols = (
            training_dataset["symbol"].tolist()
        )
        return self

    def predict(
        self,
        dataset: pd.DataFrame,
    ) -> pd.Series | list[int]:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        if self.predictions is not None:
            return self.predictions

        return pd.Series(
            dataset["cluster_id"].to_numpy(),
            index=dataset.index,
        )


@pytest.fixture
def training_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOG"],
            "snapshot_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-08",
                    "2024-01-15",
                ]
            ),
            "target_date": pd.to_datetime(
                [
                    "2024-01-29",
                    "2024-02-05",
                    "2024-02-12",
                ]
            ),
            "cluster_id": [1, 2, 1],
            "target_cluster_id": [1, 2, 2],
        }
    )


@pytest.fixture
def test_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOG"],
            "snapshot_date": pd.to_datetime(
                [
                    "2024-03-01",
                    "2024-03-08",
                    "2024-03-15",
                ]
            ),
            "target_date": pd.to_datetime(
                [
                    "2024-03-29",
                    "2024-04-05",
                    "2024-04-12",
                ]
            ),
            "cluster_id": [1, 2, 1],
            "target_cluster_id": [1, 1, 2],
        },
        index=[10, 20, 30],
    )


@pytest.fixture
def metadata() -> EvaluationRunMetadata:
    return EvaluationRunMetadata(
        run_id="test-run-001",
        model_name="DummyModel",
        model_version="1.0",
        investment_horizon="medium",
        forecast_horizon_weeks=4,
        dataset_version="test-dataset-001",
        training_start_date=datetime(2024, 1, 1),
        training_end_date=datetime(2024, 2, 12),
        test_start_date=datetime(2024, 3, 1),
        test_end_date=datetime(2024, 3, 15),
        evaluation_timestamp=datetime(2024, 4, 20),
        model_configuration={},
    )


def test_pipeline_returns_evaluation_run(
    training_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    metadata: EvaluationRunMetadata,
) -> None:
    model = DummyForecastingModel()

    result = EvaluationPipeline().run(
        model=model,
        training_dataset=training_dataset,
        test_dataset=test_dataset,
        metadata=metadata,
    )

    assert result.metadata == metadata
    assert result.report_path is None
    assert result.evaluation.overall_metrics["sample_count"] == 3
    assert len(result.prediction_result.predictions) == 3


def test_pipeline_fits_model_using_training_data_only(
    training_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    metadata: EvaluationRunMetadata,
) -> None:
    model = DummyForecastingModel()

    EvaluationPipeline().run(
        model=model,
        training_dataset=training_dataset,
        test_dataset=test_dataset,
        metadata=metadata,
    )

    assert model.is_fitted is True
    assert model.training_symbols == ["AAPL", "MSFT", "GOOG"]
    assert "TEST" not in model.training_symbols


def test_pipeline_preserves_prediction_order(
    training_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    metadata: EvaluationRunMetadata,
) -> None:
    model = DummyForecastingModel()

    result = EvaluationPipeline().run(
        model=model,
        training_dataset=training_dataset,
        test_dataset=test_dataset,
        metadata=metadata,
    )

    predictions = result.prediction_result.predictions

    assert predictions["symbol"].tolist() == [
        "AAPL",
        "MSFT",
        "GOOG",
    ]

    assert predictions["predicted_cluster_id"].tolist() == [
        1,
        2,
        1,
    ]


def test_pipeline_rejects_wrong_prediction_count(
    training_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    metadata: EvaluationRunMetadata,
) -> None:
    model = DummyForecastingModel(
        predictions=[1, 2],
    )

    with pytest.raises(ValueError, match="Prediction count"):
        EvaluationPipeline().run(
            model,
            training_dataset,
            test_dataset,
            metadata,
        )


def test_pipeline_rejects_mismatched_prediction_index(
    training_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    metadata: EvaluationRunMetadata,
) -> None:
    model = DummyForecastingModel(
        predictions=pd.Series(
            [1, 2, 1],
            index=[0, 1, 2],
        ),
    )

    with pytest.raises(ValueError, match="Prediction index"):
        EvaluationPipeline().run(
            model,
            training_dataset,
            test_dataset,
            metadata,
        )


def test_pipeline_rejects_missing_predictions(
    training_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    metadata: EvaluationRunMetadata,
) -> None:
    model = DummyForecastingModel(
        predictions=[1, None, 2],
    )

    with pytest.raises(ValueError, match="missing values"):
        EvaluationPipeline().run(
            model,
            training_dataset,
            test_dataset,
            metadata,
        )


def test_pipeline_rejects_overlapping_training_and_test_periods(
    training_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    metadata: EvaluationRunMetadata,
) -> None:
    overlapping_training = training_dataset.copy()

    overlapping_training.loc[
        2,
        "snapshot_date",
    ] = pd.Timestamp("2024-02-02")

    overlapping_training.loc[
        2,
        "target_date",
    ] = pd.Timestamp("2024-03-01")

    with pytest.raises(
        ValueError,
        match="Training targets must end before",
    ):
        EvaluationPipeline().run(
            DummyForecastingModel(),
            overlapping_training,
            test_dataset,
            metadata,
        )


def test_pipeline_rejects_incorrect_forecast_horizon(
    training_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    metadata: EvaluationRunMetadata,
) -> None:
    invalid_test = test_dataset.copy()

    invalid_test.loc[
        10,
        "target_date",
    ] = pd.Timestamp("2024-03-22")

    with pytest.raises(
        ValueError,
        match="forecast horizon",
    ):
        EvaluationPipeline().run(
            DummyForecastingModel(),
            training_dataset,
            invalid_test,
            metadata,
        )


def test_pipeline_rejects_empty_training_dataset(
    training_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    metadata: EvaluationRunMetadata,
) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        EvaluationPipeline().run(
            DummyForecastingModel(),
            training_dataset.iloc[0:0],
            test_dataset,
            metadata,
        )


def test_pipeline_rejects_missing_required_columns(
    training_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    metadata: EvaluationRunMetadata,
) -> None:
    invalid_test = test_dataset.drop(
        columns=["target_cluster_id"]
    )

    with pytest.raises(ValueError, match="missing columns"):
        EvaluationPipeline().run(
            DummyForecastingModel(),
            training_dataset,
            invalid_test,
            metadata,
        )
