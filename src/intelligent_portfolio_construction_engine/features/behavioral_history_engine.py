
from pathlib import Path

import pandas as pd

from intelligent_portfolio_construction_engine.features.behavioral_history import (
    cluster_and_save_behavioral_history,
    combine_behavioral_snapshots,
)
from intelligent_portfolio_construction_engine.features.behavioral_snapshots import (
    build_behavioral_snapshots,
)
from intelligent_portfolio_construction_engine.features.historical_context import (
    HistoricalContextEngine,
)
from intelligent_portfolio_construction_engine.config.benchmarks import (
    get_benchmark,
)


class BehavioralHistoryEngine:
    def __init__(self, settings, provider, output_dir: Path):
        self.settings = settings
        self.provider = provider
        self.output_dir = output_dir

        self.context_engine = HistoricalContextEngine(
            settings=settings,
            provider=provider,
        )

    def build_history(self, assets: list[dict]) -> pd.DataFrame:
        start_date = self.settings.BEHAVIORAL_HISTORY_START_DATE
        end_date = self.settings.BEHAVIORAL_HISTORY_END_DATE

        if end_date is None:
            end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

        frequency = self.settings.BEHAVIORAL_SNAPSHOT_FREQUENCY
        snapshots = []

        for asset in assets:
            symbol = asset["symbol"]
            asset_class = asset["asset_class"]

            benchmark = get_benchmark(asset_class)

            asset_df = self.provider.get_data(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
            )

            benchmark_df = self.provider.get_data(
                symbols=[benchmark.symbol],
                start_date=start_date,
                end_date=end_date,
            )

            historical_series = self.context_engine.build_feature_series(
                asset_df=asset_df,
                benchmark_df=benchmark_df,
            )

            asset_snapshots = build_behavioral_snapshots(
                historical_series=historical_series,
                symbol=symbol,
                asset_class=asset_class,
                frequency=frequency,
            )

            snapshots.append(asset_snapshots)

        history = combine_behavioral_snapshots(snapshots)

        if history.empty:
            raise ValueError("No behavioral snapshots were generated.")

        history.to_csv(
            self.output_dir / "behavioral_snapshots.csv",
            index=False,
        )

        clustered_history = cluster_and_save_behavioral_history(
            snapshots=history,
            settings=self.settings,
            output_dir=self.output_dir,
        )

        return clustered_history
