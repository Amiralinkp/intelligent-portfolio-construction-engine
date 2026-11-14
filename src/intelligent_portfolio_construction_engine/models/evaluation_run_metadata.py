
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class EvaluationRunMetadata:
    run_id: str
    model_name: str
    model_version: str
    investment_horizon: str
    forecast_horizon_weeks: int
    dataset_version: str
    training_start_date: datetime
    training_end_date: datetime
    test_start_date: datetime
    test_end_date: datetime
    evaluation_timestamp: datetime
    model_configuration: dict[str, Any]

    def __post_init__(self) -> None:
        string_fields = {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "investment_horizon": self.investment_horizon,
            "dataset_version": self.dataset_version,
        }

        for field_name, value in string_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be a non-empty string."
                )

        if (
            not isinstance(self.forecast_horizon_weeks, int)
            or isinstance(self.forecast_horizon_weeks, bool)
            or self.forecast_horizon_weeks < 1
        ):
            raise ValueError(
                "Forecast horizon must be a positive integer."
            )

        date_fields = (
            "training_start_date",
            "training_end_date",
            "test_start_date",
            "test_end_date",
            "evaluation_timestamp",
        )

        date_values = []

        for field_name in date_fields:
            value = getattr(self, field_name)

            if not isinstance(value, datetime):
                raise TypeError(
                    f"{field_name} must be a datetime."
                )

            date_values.append(value)

        timezone_awareness = {
            value.utcoffset() is not None
            for value in date_values
        }

        if len(timezone_awareness) > 1:
            raise ValueError(
                "All metadata dates must use consistent "
                "timezone awareness."
            )

        if self.training_start_date > self.training_end_date:
            raise ValueError(
                "Training start date must not exceed training end date."
            )

        if self.test_start_date > self.test_end_date:
            raise ValueError(
                "Test start date must not exceed test end date."
            )

        if self.training_end_date >= self.test_start_date:
            raise ValueError(
                "Training period must end before the test period begins."
            )

        if not isinstance(self.model_configuration, dict):
            raise TypeError(
                "Model configuration must be a dictionary."
            )
