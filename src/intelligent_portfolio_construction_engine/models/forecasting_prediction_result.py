from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_bool_dtype


@dataclass(frozen=True)
class ForecastingPredictionResult:
    predictions: pd.DataFrame

    def __post_init__(self) -> None:
        if not isinstance(self.predictions, pd.DataFrame):
            raise TypeError(
                "Predictions must be a pandas DataFrame."
            )

        required_columns = {
            "symbol",
            "snapshot_date",
            "target_date",
            "current_cluster_id",
            "actual_cluster_id",
            "predicted_cluster_id",
            "is_correct",
        }

        missing_columns = (
            required_columns - set(self.predictions.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing required prediction columns: "
                f"{sorted(missing_columns)}"
            )

        if self.predictions[list(required_columns)].isna().any().any():
            raise ValueError(
                "Required prediction fields cannot contain missing values."
            )

        if not is_bool_dtype(self.predictions["is_correct"]):
            raise TypeError(
                "The is_correct column must contain boolean values."
            )

        expected_correctness = (
            self.predictions["actual_cluster_id"]
            == self.predictions["predicted_cluster_id"]
        )

        if not self.predictions["is_correct"].equals(
            expected_correctness
        ):
            raise ValueError(
                "The is_correct column does not match actual "
                "and predicted cluster IDs."
            )