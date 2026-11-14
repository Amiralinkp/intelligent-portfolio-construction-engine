
import pandas as pd


class CurrentStatePersistenceBaseline:
    REQUIRED_COLUMNS = {"cluster_id"}

    def __init__(self) -> None:
        self._is_fitted = False

    def fit(self, training_dataset: pd.DataFrame) -> "CurrentStatePersistenceBaseline":
        self._validate_dataset(training_dataset)

        if training_dataset.empty:
            raise ValueError(
                "Training dataset cannot be empty.")

        self._is_fitted = True

        return self

    def predict(self, dataset: pd.DataFrame) -> pd.Series:
        if not self._is_fitted:
            raise RuntimeError("The baseline must be fitted before prediction.")

        self._validate_dataset(dataset)

        predictions = dataset["cluster_id"].copy()

        predictions.name = "predicted_cluster_id"

        return predictions

    @classmethod
    def _validate_dataset(cls, dataset: pd.DataFrame) -> None:
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(
                "Dataset must be a pandas DataFrame."
            )

        missing_columns = (
            cls.REQUIRED_COLUMNS - set(dataset.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        if dataset["cluster_id"].isna().any():
            raise ValueError(
                "Current cluster IDs cannot contain missing values."
            )
