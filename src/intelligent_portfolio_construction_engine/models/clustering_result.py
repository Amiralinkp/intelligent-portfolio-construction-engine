from dataclasses import dataclass
from typing import Any
import pandas as pd


@dataclass(frozen=True)
class ClusteringResult:
    labels: pd.Series
    model_name: str
    estimator: Any | None = None