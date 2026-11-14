
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "volatility",
    "current_drawdown",
    "cagr",
    "momentum_factor",
    "beta",
    "sharpe_ratio",
    "sortino_ratio",
]

METADATA_COLUMNS = [
    "symbol",
    "snapshot_date",
    "target_date",
    "model_version",
    "target_model_version",
]

DATASET_COLUMNS = [
    *METADATA_COLUMNS,
    "cluster_id",
    *FEATURE_COLUMNS,
    "target_cluster_id",
]


class ForecastingDatasetBuilder:
    def __init__(
        self,
        horizon_weeks: int = 4,
        expected_interval: str | pd.Timedelta = "7D",
    ) -> None:
        if (
            not isinstance(horizon_weeks, int)
            or isinstance(horizon_weeks, bool)
            or horizon_weeks < 1
        ):
            raise ValueError("Horizon must be a positive integer.")

        interval = pd.Timedelta(expected_interval)

        if interval <= pd.Timedelta(0):
            raise ValueError("Expected interval must be positive.")

        self.horizon = horizon_weeks
        self.expected_interval = interval

    def build(
        self,
        behavioral_history: pd.DataFrame,
    ) -> pd.DataFrame:
        history = self._validate_history(behavioral_history)

        records = []

        for symbol, asset_history in history.groupby("symbol"):
            asset_history = asset_history.set_index("snapshot_date")

            available_dates = set(asset_history.index)

            for snapshot_date, source in asset_history.iterrows():
                target_date = (
                    snapshot_date
                    + self.horizon * self.expected_interval
                )

                required_dates = {
                    snapshot_date + step * self.expected_interval
                    for step in range(1, self.horizon + 1)
                }

                if not required_dates.issubset(available_dates):
                    continue

                target = asset_history.loc[target_date]

                record = {
                    "symbol": symbol,
                    "snapshot_date": snapshot_date,
                    "target_date": target_date,
                    "model_version": source["model_version"],
                    "target_model_version": target["model_version"],
                    "cluster_id": source["cluster_id"],
                    "target_cluster_id": target["cluster_id"],
                }

                record.update(
                    {
                        feature: source[feature]
                        for feature in FEATURE_COLUMNS
                    }
                )

                records.append(record)

        return pd.DataFrame(
            records,
            columns=DATASET_COLUMNS,
        )

    @staticmethod
    def _validate_history(
        behavioral_history: pd.DataFrame,
    ) -> pd.DataFrame:
        required_columns = {
            *METADATA_COLUMNS[:2],
            "model_version",
            "cluster_id",
            *FEATURE_COLUMNS,
        }

        missing_columns = required_columns - set(
            behavioral_history.columns
        )

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        history = behavioral_history.copy()

        history["snapshot_date"] = pd.to_datetime(
            history["snapshot_date"]
        )

        if history["snapshot_date"].isna().any():
            raise ValueError("Snapshot dates cannot be missing.")

        if history["symbol"].isna().any():
            raise ValueError("Asset symbols cannot be missing.")

        if history["model_version"].isna().any():
            raise ValueError("Model versions cannot be missing.")

        if history["cluster_id"].isna().any():
            raise ValueError("Cluster IDs cannot be missing.")

        if history.duplicated(
            ["symbol", "snapshot_date"]
        ).any():
            raise ValueError(
                "Duplicate asset snapshots detected."
            )

        try:
            feature_values = history[FEATURE_COLUMNS].to_numpy(
                dtype=float
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Forecasting features must be numeric."
            ) from exc

        if not np.isfinite(feature_values).all():
            raise ValueError(
                "Forecasting features must contain finite values."
            )

        return history.sort_values(
            ["symbol", "snapshot_date"],
            ignore_index=True,
        )
