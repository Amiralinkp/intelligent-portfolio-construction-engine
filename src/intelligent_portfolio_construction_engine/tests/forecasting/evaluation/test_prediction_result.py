
import pandas as pd
import pytest

from intelligent_portfolio_construction_engine.models.forecasting_prediction_result import (
    ForecastingPredictionResult,
)


@pytest.fixture
def valid_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOG"],
            "snapshot_date": pd.to_datetime(
                ["2024-01-01", "2024-01-08", "2024-01-15"]
            ),
            "target_date": pd.to_datetime(
                ["2024-01-29", "2024-02-05", "2024-02-12"]
            ),
            "current_cluster_id": [1, 2, 3],
            "actual_cluster_id": [1, 3, 2],
            "predicted_cluster_id": [1, 2, 2],
            "is_correct": [True, False, True],
        }
    )


def test_valid_predictions_are_accepted(
    valid_predictions: pd.DataFrame,
) -> None:
    result = ForecastingPredictionResult(
        predictions=valid_predictions
    )

    pd.testing.assert_frame_equal(
        result.predictions,
        valid_predictions,
    )


def test_rejects_non_dataframe_input() -> None:
    with pytest.raises(TypeError):
        ForecastingPredictionResult(
            predictions=[1, 2, 3]
        )


def test_rejects_missing_required_columns(
    valid_predictions: pd.DataFrame,
) -> None:
    invalid = valid_predictions.drop(
        columns=["predicted_cluster_id"]
    )

    with pytest.raises(ValueError, match="Missing required"):
        ForecastingPredictionResult(predictions=invalid)


def test_rejects_missing_required_values(
    valid_predictions: pd.DataFrame,
) -> None:
    invalid = valid_predictions.copy()
    invalid.loc[0, "symbol"] = None

    with pytest.raises(ValueError, match="missing values"):
        ForecastingPredictionResult(predictions=invalid)


def test_rejects_non_boolean_correctness(
    valid_predictions: pd.DataFrame,
) -> None:
    invalid = valid_predictions.copy()
    invalid["is_correct"] = [1, 0, 1]

    with pytest.raises(TypeError, match="boolean"):
        ForecastingPredictionResult(predictions=invalid)


def test_rejects_inconsistent_correctness(
    valid_predictions: pd.DataFrame,
) -> None:
    invalid = valid_predictions.copy()
    invalid.loc[0, "is_correct"] = False

    with pytest.raises(ValueError, match="does not match"):
        ForecastingPredictionResult(predictions=invalid)
