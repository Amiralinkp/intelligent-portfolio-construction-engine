
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ForecastingEvaluationResult:
    overall_metrics: dict[str, float | int]
    class_metrics: pd.DataFrame
    confusion_matrix: pd.DataFrame
    error_analysis: dict[str, int | float]
