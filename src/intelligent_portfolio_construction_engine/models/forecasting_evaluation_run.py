
from dataclasses import dataclass
from pathlib import Path

from intelligent_portfolio_construction_engine.models.evaluation_run_metadata import (
    EvaluationRunMetadata,
)
from intelligent_portfolio_construction_engine.models.forecasting_evaluation_result import (
    ForecastingEvaluationResult,
)
from intelligent_portfolio_construction_engine.models.forecasting_prediction_result import (
    ForecastingPredictionResult,
)


@dataclass(frozen=True)
class ForecastingEvaluationRun:
    metadata: EvaluationRunMetadata
    evaluation: ForecastingEvaluationResult
    prediction_result: ForecastingPredictionResult
    report_path: Path | None = None
