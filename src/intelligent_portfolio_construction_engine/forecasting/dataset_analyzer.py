
import numpy as np
import pandas as pd

from intelligent_portfolio_construction_engine.models.forecasting_dataset_analysis import (
    ForecastingDatasetAnalysis,
)

from .dataset_builder import FEATURE_COLUMNS


class DatasetAnalyzer:
    MINIMUM_SAMPLE_SIZE = 30
    HIGH_CLASS_SHARE_THRESHOLD = 0.80
    LOW_CLASS_SHARE_THRESHOLD = 0.05

    REQUIRED_COLUMNS = {
        "symbol",
        "snapshot_date",
        "target_date",
        "model_version",
        "target_model_version",
        "cluster_id",
        "target_cluster_id",
        *FEATURE_COLUMNS,
    }

    def analyze(
        self,
        dataset: pd.DataFrame,
    ) -> ForecastingDatasetAnalysis:
        data = self._validate_dataset(dataset)

        dataset_summary = self._analyze_dataset(data)
        quality_report = self._analyze_quality(data)
        feature_statistics = self._analyze_features(data)
        target_distribution = self._analyze_target(data)

        warnings = self._generate_warnings(
            dataset_summary,
            quality_report,
            target_distribution,
        )

        recommendations = self._generate_recommendations(
            dataset_summary,
            quality_report,
            target_distribution,
            warnings,
        )

        return ForecastingDatasetAnalysis(
            dataset_summary=dataset_summary,
            quality_report=quality_report,
            feature_statistics=feature_statistics,
            target_distribution=target_distribution,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _validate_dataset(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("Dataset must be a pandas DataFrame.")

        missing_columns = self.REQUIRED_COLUMNS - set(dataset.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        return dataset.copy()

    @staticmethod
    def _analyze_dataset(
        dataset: pd.DataFrame,
    ) -> dict:
        snapshot_dates = pd.to_datetime(
            dataset["snapshot_date"],
            errors="coerce",
        )

        target_dates = pd.to_datetime(
            dataset["target_date"],
            errors="coerce",
        )

        valid_horizons = (
            target_dates - snapshot_dates
        ).dropna()

        return {
            "sample_count": len(dataset),
            "asset_count": int(dataset["symbol"].nunique()),
            "source_model_count": int(
                dataset["model_version"].nunique()
            ),
            "target_model_count": int(
                dataset["target_model_version"].nunique()
            ),
            "snapshot_start": (
                snapshot_dates.min().isoformat()
                if snapshot_dates.notna().any()
                else None
            ),
            "snapshot_end": (
                snapshot_dates.max().isoformat()
                if snapshot_dates.notna().any()
                else None
            ),
            "average_forecast_horizon_days": (
                valid_horizons.dt.total_seconds().mean() / 86400
                if not valid_horizons.empty
                else None
            ),
        }

    @staticmethod
    def _analyze_quality(
        dataset: pd.DataFrame,
    ) -> dict:
        missing_values = dataset.isna().sum()

        numeric_features = dataset[FEATURE_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )

        original_missing = dataset[FEATURE_COLUMNS].isna().sum()

        non_numeric_values = (
            numeric_features.isna() & dataset[FEATURE_COLUMNS].notna()
        ).sum()

        non_finite_values = pd.Series(
            {
                feature: int(
                    np.isinf(
                        numeric_features[feature].dropna().to_numpy(
                            dtype=float
                        )
                    ).sum()
                )
                for feature in FEATURE_COLUMNS
            }
        )

        duplicate_count = int(
            dataset.duplicated(
                ["symbol", "snapshot_date"]
            ).sum()
        )

        return {
            "total_missing_values": int(missing_values.sum()),
            "missing_values_by_column": {
                column: int(count)
                for column, count in missing_values.items()
            },
            "duplicate_asset_snapshots": duplicate_count,
            "missing_target_count": int(
                dataset["target_cluster_id"].isna().sum()
            ),
            "missing_feature_values": {
                feature: int(count)
                for feature, count in original_missing.items()
            },
            "non_numeric_feature_values": {
                feature: int(count)
                for feature, count in non_numeric_values.items()
            },
            "non_finite_feature_values": {
                feature: int(count)
                for feature, count in non_finite_values.items()
            },
        }

    
    @staticmethod
    def _analyze_features(
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        records = []

        for feature in FEATURE_COLUMNS:
            original = dataset[feature]

            numeric = pd.to_numeric(
                original,
                errors="coerce",
            )

            values = numeric.to_numpy(
                dtype=float,
                na_value=np.nan,
            )

            finite_values = values[np.isfinite(values)]

            records.append(
                {
                    "feature": feature,
                    "count": len(finite_values),
                    "mean": (
                        float(np.mean(finite_values))
                        if len(finite_values)
                        else np.nan
                    ),
                    "std": (
                        float(np.std(finite_values, ddof=1))
                        if len(finite_values) > 1
                        else np.nan
                    ),
                    "min": (
                        float(np.min(finite_values))
                        if len(finite_values)
                        else np.nan
                    ),
                    "25%": (
                        float(np.percentile(finite_values, 25))
                        if len(finite_values)
                        else np.nan
                    ),
                    "median": (
                        float(np.median(finite_values))
                        if len(finite_values)
                        else np.nan
                    ),
                    "75%": (
                        float(np.percentile(finite_values, 75))
                        if len(finite_values)
                        else np.nan
                    ),
                    "max": (
                        float(np.max(finite_values))
                        if len(finite_values)
                        else np.nan
                    ),
                    "missing_count": int(original.isna().sum()),
                    "non_numeric_count": int(
                        (numeric.isna() & original.notna()).sum()
                    ),
                    "non_finite_count": int(
                        np.isinf(values).sum()
                    ),
                }
            )

        return pd.DataFrame.from_records(records)


    @staticmethod
    def _analyze_target(
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        targets = dataset["target_cluster_id"].dropna()

        if targets.empty:
            return pd.DataFrame(
                columns=["cluster_id", "sample_count", "share"]
            )

        distribution = (
            targets.value_counts()
            .rename_axis("cluster_id")
            .reset_index(name="sample_count")
        )

        distribution["share"] = (
            distribution["sample_count"] / len(targets)
        )

        return distribution.sort_values(
            "cluster_id"
        ).reset_index(drop=True)

    def _generate_warnings(
        self,
        dataset_summary: dict,
        quality_report: dict,
        target_distribution: pd.DataFrame,
    ) -> list[str]:
        warnings = []

        sample_count = dataset_summary["sample_count"]

        if sample_count == 0:
            warnings.append(
                "[HIGH] Dataset is empty. No forecasting samples are available."
            )
            return warnings

        if sample_count < self.MINIMUM_SAMPLE_SIZE:
            warnings.append(
                "[MEDIUM] Dataset contains fewer than 30 samples. "
                "Model evaluation may be unreliable."
            )

        if quality_report["total_missing_values"] > 0:
            warnings.append(
                "[HIGH] Missing values were detected in the dataset."
            )

        if quality_report["duplicate_asset_snapshots"] > 0:
            warnings.append(
                "[HIGH] Duplicate asset snapshots were detected."
            )

        invalid_feature_values = (
            sum(quality_report["non_numeric_feature_values"].values())
            + sum(quality_report["non_finite_feature_values"].values())
        )

        if invalid_feature_values > 0:
            warnings.append(
                "[HIGH] Invalid or non-finite feature values were detected."
            )

        if target_distribution.empty:
            warnings.append(
                "[HIGH] No valid target labels are available."
            )
            return warnings

        if len(target_distribution) < 2:
            warnings.append(
                "[HIGH] Only one target class is present. "
                "Classification between states cannot be evaluated."
            )

        largest_share = float(target_distribution["share"].max())
        smallest_share = float(target_distribution["share"].min())

        if largest_share >= self.HIGH_CLASS_SHARE_THRESHOLD:
            warnings.append(
                "[MEDIUM] Target distribution is highly imbalanced. "
                "The majority class represents at least 80% of samples."
            )

        if (
            len(target_distribution) > 1
            and smallest_share < self.LOW_CLASS_SHARE_THRESHOLD
        ):
            warnings.append(
                "[MEDIUM] At least one target class represents less "
                "than 5% of valid samples."
            )

        return warnings

    @staticmethod
    def _generate_recommendations(
        dataset_summary: dict,
        quality_report: dict,
        target_distribution: pd.DataFrame,
        warnings: list[str],
    ) -> list[str]:
        recommendations = []

        if dataset_summary["sample_count"] == 0:
            recommendations.append(
                "Review the source behavioral history and forecasting "
                "horizon before attempting model training."
            )
            return recommendations

        if quality_report["total_missing_values"] > 0:
            recommendations.append(
                "Investigate missing values and decide how each affected "
                "field should be handled before model training."
            )

        if quality_report["duplicate_asset_snapshots"] > 0:
            recommendations.append(
                "Review duplicate symbol and snapshot-date combinations "
                "in the source dataset."
            )

        if (
            sum(quality_report["non_numeric_feature_values"].values())
            + sum(quality_report["non_finite_feature_values"].values())
        ) > 0:
            recommendations.append(
                "Inspect invalid feature values and their upstream "
                "calculations before training."
            )

        if len(target_distribution) < 2:
            recommendations.append(
                "Check whether the selected period and forecast horizon "
                "contain enough behavioral state transitions."
            )
        elif warnings and any(
            "imbalanced" in warning.lower()
            or "less than 5%" in warning.lower()
            for warning in warnings
        ):
            recommendations.append(
                "Use class-sensitive evaluation metrics and compare "
                "model performance against simple baselines."
            )

        if not recommendations:
            recommendations.append(
                "No major dataset quality issues were detected. "
                "Proceed to deeper validation and baseline evaluation."
            )

        return recommendations
