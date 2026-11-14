
from collections.abc import Sequence
from typing import Protocol, Self

import numpy as np
import pandas as pd

from intelligent_portfolio_construction_engine.forecasting.evaluation.classification_evaluator import (
    ClassificationEvaluator,
)
from intelligent_portfolio_construction_engine.models.evaluation_run_metadata import (
    EvaluationRunMetadata,
)
from intelligent_portfolio_construction_engine.models.forecasting_evaluation_run import (
    ForecastingEvaluationRun,
)
from intelligent_portfolio_construction_engine.models.forecasting_prediction_result import (
    ForecastingPredictionResult,
)


class ForecastingModel(Protocol):
    def fit(
        self,
        training_dataset: pd.DataFrame,
    ) -> Self:
        ...

    def predict(
        self,
        dataset: pd.DataFrame,
    ) -> pd.Series | Sequence[object] | np.ndarray:
        ...


class EvaluationPipeline:
    REQUIRED_DATASET_COLUMNS = {
        "symbol",
        "snapshot_date",
        "target_date",
        "cluster_id",
        "target_cluster_id",
    }

    def __init__(
        self,
        evaluator: ClassificationEvaluator | None = None,
    ) -> None:
        self.evaluator = evaluator or ClassificationEvaluator()

    def run(
        self,
        model: ForecastingModel,
        training_dataset: pd.DataFrame,
        test_dataset: pd.DataFrame,
        metadata: EvaluationRunMetadata,
    ) -> ForecastingEvaluationRun:
        self._validate_datasets(
            training_dataset,
            test_dataset,
            metadata,
        )

        model.fit(training_dataset)

        raw_predictions = model.predict(test_dataset)

        if len(raw_predictions) != len(test_dataset):
            raise ValueError(
                "Prediction count must match test dataset length."
            )

        if isinstance(raw_predictions, pd.Series):
            if not raw_predictions.index.equals(test_dataset.index):
                raise ValueError(
                    "Prediction index must match test dataset index."
                )

            predicted = raw_predictions.reset_index(drop=True)
        else:
            predicted = pd.Series(raw_predictions)

        if predicted.isna().any():
            raise ValueError(
                "Model predictions cannot contain missing values."
            )

        test = test_dataset.reset_index(drop=True)

        actual = test["target_cluster_id"]

        prediction_frame = pd.DataFrame(
            {
                "symbol": test["symbol"],
                "snapshot_date": test["snapshot_date"],
                "target_date": test["target_date"],
                "current_cluster_id": test["cluster_id"],
                "actual_cluster_id": actual,
                "predicted_cluster_id": predicted,
                "is_correct": actual.eq(predicted),
            }
        )

        known_labels = pd.unique(
            pd.concat(
                [
                    training_dataset["cluster_id"],
                    training_dataset["target_cluster_id"],
                ],
                ignore_index=True,
            )
        ).tolist()

        evaluation_labels = list(
            dict.fromkeys(
                [
                    *known_labels,
                    *pd.unique(actual).tolist(),
                    *pd.unique(predicted).tolist(),
                ]
            )
        )

        evaluation = self.evaluator.evaluate(
            y_true=actual,
            y_pred=predicted,
            labels=evaluation_labels,
            known_labels=known_labels,
        )

        prediction_result = ForecastingPredictionResult(
            predictions=prediction_frame,
        )

        return ForecastingEvaluationRun(
            metadata=metadata,
            evaluation=evaluation,
            prediction_result=prediction_result,
        )

    @classmethod
    def _validate_datasets(
        cls,
        training_dataset: pd.DataFrame,
        test_dataset: pd.DataFrame,
        metadata: EvaluationRunMetadata,
    ) -> None:
        if (
            not isinstance(metadata.forecast_horizon_weeks, int)
            or isinstance(metadata.forecast_horizon_weeks, bool)
            or metadata.forecast_horizon_weeks < 1
        ):
            raise ValueError(
                "Forecast horizon must be a positive integer."
            )

        for name, dataset in (
            ("Training", training_dataset),
            ("Test", test_dataset),
        ):
            if not isinstance(dataset, pd.DataFrame):
                raise TypeError(
                    f"{name} dataset must be a DataFrame."
                )

            missing_columns = (
                cls.REQUIRED_DATASET_COLUMNS
                - set(dataset.columns)
            )

            if missing_columns:
                raise ValueError(
                    f"{name} dataset is missing columns: "
                    f"{sorted(missing_columns)}"
                )

            if dataset.empty:
                raise ValueError(
                    f"{name} dataset cannot be empty."
                )

            if dataset[list(cls.REQUIRED_DATASET_COLUMNS)].isna().any().any():
                raise ValueError(
                    f"{name} dataset contains missing required values."
                )

            snapshot_dates = pd.to_datetime(
                dataset["snapshot_date"],
                errors="raise",
            )

            target_dates = pd.to_datetime(
                dataset["target_date"],
                errors="raise",
            )

            if snapshot_dates.isna().any() or target_dates.isna().any():
                raise ValueError(
                    f"{name} dataset contains invalid dates."
                )

            if (target_dates <= snapshot_dates).any():
                raise ValueError(
                    f"{name} dataset contains invalid target dates."
                )

            expected_duration = pd.Timedelta(
                weeks=metadata.forecast_horizon_weeks
            )

            actual_durations = target_dates - snapshot_dates

            if not actual_durations.eq(expected_duration).all():
                raise ValueError(
                    f"{name} dataset does not match the configured "
                    "forecast horizon."
                )

        training_target_end = pd.to_datetime(
            training_dataset["target_date"]
        ).max()

        test_snapshot_start = pd.to_datetime(
            test_dataset["snapshot_date"]
        ).min()

        if training_target_end >= test_snapshot_start:
            raise ValueError(
                "Training targets must end before the test period begins."
            )
