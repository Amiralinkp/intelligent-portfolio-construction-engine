
from pathlib import Path

import pandas as pd

from intelligent_portfolio_construction_engine.forecasting.dataset_analyzer import (
    DatasetAnalyzer,
)
from intelligent_portfolio_construction_engine.forecasting.dataset_builder import (
    ForecastingDatasetBuilder,
)
from intelligent_portfolio_construction_engine.forecasting.reporting.dataset_reporter import (
    DatasetReporter,
)
from intelligent_portfolio_construction_engine.models.forecasting_dataset_analysis import (
    ForecastingDatasetAnalysis,
)


class ForecastingDatasetPipeline:
    def __init__(
        self,
        horizon_weeks: int = 4,
        expected_interval: str | pd.Timedelta = "7D",
        output_dir: str | Path = "artifacts/forecasting/dataset_analysis",
    ) -> None:
        self.dataset_builder = ForecastingDatasetBuilder(
            horizon_weeks=horizon_weeks,
            expected_interval=expected_interval,
        )
        self.dataset_analyzer = DatasetAnalyzer()
        self.dataset_reporter = DatasetReporter(
            output_dir=output_dir,
        )

    def run(
        self,
        behavioral_history: pd.DataFrame,
    ) -> tuple[pd.DataFrame, ForecastingDatasetAnalysis, Path]:
        dataset = self.dataset_builder.build(
            behavioral_history,
        )

        analysis = self.dataset_analyzer.analyze(
            dataset,
        )

        report_path = self.dataset_reporter.generate(
            analysis=analysis,
            dataset=dataset,
        )

        return dataset, analysis, report_path
