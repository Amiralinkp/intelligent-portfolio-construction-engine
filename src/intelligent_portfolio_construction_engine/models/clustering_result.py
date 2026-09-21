from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class ClusteringResult:
    
    labels: pd.Series
    model_name: str