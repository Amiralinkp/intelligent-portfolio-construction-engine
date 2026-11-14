from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class ForecastingDatasetAnalysis:
    dataset_summary: dict
    quality_report: dict
    feature_statistics: pd.DataFrame
    target_distribution: pd.DataFrame
    warnings: list[str]
    recommendations: list[str]