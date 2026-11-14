
from datetime import datetime, timezone

import pytest

from intelligent_portfolio_construction_engine.models.evaluation_run_metadata import (
    EvaluationRunMetadata,
)


@pytest.fixture
def valid_metadata() -> EvaluationRunMetadata:
    return EvaluationRunMetadata(
        run_id="run-001",
        model_name="MarkovBaseline",
        model_version="1.0",
        investment_horizon="medium",
        forecast_horizon_weeks=4,
        dataset_version="dataset-001",
        training_start_date=datetime(2024, 1, 1),
        training_end_date=datetime(2024, 2, 1),
        test_start_date=datetime(2024, 3, 1),
        test_end_date=datetime(2024, 4, 1),
        evaluation_timestamp=datetime(2024, 4, 10),
        model_configuration={"alpha": 1.0},
    )


def test_valid_metadata_is_accepted(
    valid_metadata: EvaluationRunMetadata,
) -> None:
    assert valid_metadata.run_id == "run-001"
    assert valid_metadata.forecast_horizon_weeks == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", ""),
        ("model_name", " "),
        ("model_version", ""),
        ("investment_horizon", ""),
        ("dataset_version", ""),
    ],
)
def test_rejects_empty_string_fields(
    valid_metadata: EvaluationRunMetadata,
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValueError):
        EvaluationRunMetadata(
            **{
                **valid_metadata.__dict__,
                field: value,
            }
        )


@pytest.mark.parametrize(
    "horizon",
    [0, -1, True, 1.5],
)
def test_rejects_invalid_forecast_horizon(
    valid_metadata: EvaluationRunMetadata,
    horizon: int | float | bool,
) -> None:
    with pytest.raises(ValueError):
        EvaluationRunMetadata(
            **{
                **valid_metadata.__dict__,
                "forecast_horizon_weeks": horizon,
            }
        )


def test_rejects_training_start_after_end(
    valid_metadata: EvaluationRunMetadata,
) -> None:
    with pytest.raises(ValueError, match="Training start date"):
        EvaluationRunMetadata(
            **{
                **valid_metadata.__dict__,
                "training_start_date": datetime(2024, 2, 2),
            }
        )


def test_rejects_test_start_after_end(
    valid_metadata: EvaluationRunMetadata,
) -> None:
    with pytest.raises(ValueError, match="Test start date"):
        EvaluationRunMetadata(
            **{
                **valid_metadata.__dict__,
                "test_start_date": datetime(2024, 4, 2),
            }
        )


def test_rejects_overlapping_periods(
    valid_metadata: EvaluationRunMetadata,
) -> None:
    with pytest.raises(ValueError, match="Training period"):
        EvaluationRunMetadata(
            **{
                **valid_metadata.__dict__,
                "training_end_date": datetime(2024, 3, 1),
            }
        )


def test_rejects_inconsistent_timezone_awareness(
    valid_metadata: EvaluationRunMetadata,
) -> None:
    with pytest.raises(ValueError, match="timezone awareness"):
        EvaluationRunMetadata(
            **{
                **valid_metadata.__dict__,
                "training_start_date": datetime(
                    2024, 1, 1, tzinfo=timezone.utc
                ),
            }
        )


def test_accepts_consistent_timezone_aware_dates() -> None:
    utc = timezone.utc

    metadata = EvaluationRunMetadata(
        run_id="run-utc",
        model_name="TestModel",
        model_version="1.0",
        investment_horizon="medium",
        forecast_horizon_weeks=4,
        dataset_version="dataset-utc",
        training_start_date=datetime(2024, 1, 1, tzinfo=utc),
        training_end_date=datetime(2024, 2, 1, tzinfo=utc),
        test_start_date=datetime(2024, 3, 1, tzinfo=utc),
        test_end_date=datetime(2024, 4, 1, tzinfo=utc),
        evaluation_timestamp=datetime(2024, 4, 10, tzinfo=utc),
        model_configuration={},
    )

    assert metadata.run_id == "run-utc"


def test_rejects_non_datetime_values(
    valid_metadata: EvaluationRunMetadata,
) -> None:
    with pytest.raises(TypeError, match="must be a datetime"):
        EvaluationRunMetadata(
            **{
                **valid_metadata.__dict__,
                "training_start_date": "2024-01-01",
            }
        )


def test_rejects_non_dictionary_configuration(
    valid_metadata: EvaluationRunMetadata,
) -> None:
    with pytest.raises(TypeError, match="dictionary"):
        EvaluationRunMetadata(
            **{
                **valid_metadata.__dict__,
                "model_configuration": ["alpha", 1.0],
            }
        )
