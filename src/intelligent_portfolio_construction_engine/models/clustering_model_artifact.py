from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClusteringModelArtifact:
    estimator: Any
    scaler: Any
    feature_columns: tuple[str, ...]
    model_name: str
    n_clusters: int
    training_cutoff: pd.Timestamp
    model_version: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ValueError("Feature columns cannot be empty.")

        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("Feature columns must be unique.")

        if self.n_clusters < 2:
            raise ValueError("The number of clusters must be at least 2.")

        if not self.model_version:
            raise ValueError("Model version cannot be empty.")

        object.__setattr__(
            self,
            "training_cutoff",
            pd.Timestamp(self.training_cutoff),
        )

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        if data.empty:
            raise ValueError("Input data cannot be empty.")

        missing = set(self.feature_columns).difference(data.columns)
        extra = set(data.columns).difference(self.feature_columns)

        if missing or extra:
            raise ValueError(
                f"Feature mismatch. Missing: {sorted(missing)}; "
                f"unexpected: {sorted(extra)}."
            )

        features = data.loc[:, list(self.feature_columns)]

        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError("Input features must contain finite numeric values.")

        scaled_values = self.scaler.transform(features)

        scaled_features = pd.DataFrame(
            scaled_values,
            columns=self.feature_columns,
            index=data.index,
        )

        labels = self.estimator.predict(scaled_features)

        return pd.Series(
            labels,
            index=data.index,
            name="cluster",
        )