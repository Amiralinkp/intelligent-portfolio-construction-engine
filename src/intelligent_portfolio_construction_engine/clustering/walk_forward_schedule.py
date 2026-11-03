from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TrainingOrigin:
    version: int
    training_cutoff: pd.Timestamp
    next_training_cutoff: pd.Timestamp | None


class WalkForwardSchedule:

    def __init__(
        self,
        retraining_interval_weeks: int = 52,
        minimum_training_weeks: int = 104,
    ) -> None:
        if retraining_interval_weeks < 1:
            raise ValueError(
                "Retraining interval must be at least one week."
            )

        if minimum_training_weeks < 1:
            raise ValueError(
                "Minimum training period must be at least one week."
            )

        self.retraining_interval_weeks = retraining_interval_weeks
        self.minimum_training_weeks = minimum_training_weeks

    def generate(
        self,
        snapshot_dates: pd.Series,
    ) -> list[TrainingOrigin]:
        if snapshot_dates.empty:
            raise ValueError("Snapshot dates cannot be empty.")

        dates = pd.DatetimeIndex(
            pd.to_datetime(snapshot_dates).dropna().unique()
        ).sort_values()

        if len(dates) < 2:
            raise ValueError(
                "At least two distinct snapshot dates are required."
            )

        first_date = dates[0]
        last_date = dates[-1]

        first_cutoff = (
            first_date
            + pd.Timedelta(weeks=self.minimum_training_weeks)
        )

        cutoffs = []
        current_cutoff = first_cutoff

        while current_cutoff < last_date:
            available_dates = dates[dates <= current_cutoff]

            if len(available_dates) == 0:
                break

            effective_cutoff = available_dates[-1]

            if not cutoffs or effective_cutoff > cutoffs[-1]:
                cutoffs.append(effective_cutoff)

            current_cutoff += pd.Timedelta(
                weeks=self.retraining_interval_weeks
            )

        origins = []

        for index, cutoff in enumerate(cutoffs):
            next_cutoff = (
                cutoffs[index + 1]
                if index + 1 < len(cutoffs)
                else None
            )

            origins.append(
                TrainingOrigin(
                    version=index + 1,
                    training_cutoff=cutoff,
                    next_training_cutoff=next_cutoff,
                )
            )

        return origins