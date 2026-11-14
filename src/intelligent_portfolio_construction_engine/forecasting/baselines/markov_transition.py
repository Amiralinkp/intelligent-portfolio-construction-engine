import math
import pandas as pd


class MarkovTransitionBaseline:
    REQUIRED_COLUMNS = {
        "cluster_id",
        "target_cluster_id",
    }

    def __init__(
        self,
        alpha: float = 1.0,
    ) -> None:
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Smoothing alpha must be a finite positive number.")

        self.alpha = alpha
        self._is_fitted = False
        self._transition_matrix = pd.DataFrame()
        self._observed_source_states = set()

    def fit(
        self,
        training_dataset: pd.DataFrame,
    ) -> "MarkovTransitionBaseline":
        self._validate_dataset(training_dataset)

        if training_dataset.empty:
            raise ValueError(
                "Training dataset cannot be empty."
            )

        states = pd.Index(
            pd.unique(
                pd.concat(
                    [
                        training_dataset["cluster_id"],
                        training_dataset["target_cluster_id"],
                    ],
                    ignore_index=True,
                )
            )
        )

        transition_counts = pd.crosstab(
            training_dataset["cluster_id"],
            training_dataset["target_cluster_id"],
        )

        transition_counts = transition_counts.reindex(
            index=states,
            columns=states,
            fill_value=0,
        )

        smoothed_counts = transition_counts + self.alpha

        self._transition_matrix = smoothed_counts.div(
            smoothed_counts.sum(axis=1),
            axis=0,
        )

        self._observed_source_states = set(
            training_dataset["cluster_id"].unique()
        )

        self._is_fitted = True

        return self

    def predict(
        self,
        dataset: pd.DataFrame,
    ) -> pd.Series:
        if not self._is_fitted:
            raise RuntimeError(
                "The baseline must be fitted before prediction."
            )

        self._validate_dataset(
            dataset,
            require_target=False,
        )

        predictions = []

        for current_cluster in dataset["cluster_id"]:
            if current_cluster not in self._observed_source_states:
                predictions.append(current_cluster)
                continue

            probabilities = self._transition_matrix.loc[
                current_cluster
            ]

            predictions.append(
                probabilities.idxmax()
            )

        return pd.Series(
            predictions,
            index=dataset.index,
            name="predicted_cluster_id",
        )

    @classmethod
    def _validate_dataset(cls, dataset: pd.DataFrame, require_target: bool = True) -> None:
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(
                "Dataset must be a pandas DataFrame."
            )

        required_columns = (
            cls.REQUIRED_COLUMNS
            if require_target
            else {"cluster_id"}
        )

        missing_columns = (
            required_columns - set(dataset.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        if dataset["cluster_id"].isna().any():
            raise ValueError(
                "Current cluster IDs cannot contain missing values."
            )

        if (
            require_target
            and dataset["target_cluster_id"].isna().any()
        ):
            raise ValueError(
                "Target cluster IDs cannot contain missing values."
            )
