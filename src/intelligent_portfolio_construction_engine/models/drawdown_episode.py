from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class DrawdownEpisode:
    
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: Optional[pd.Timestamp]

    peak_price: float
    trough_price: float

    magnitude: float
    drawdown_duration: int

    recovery_25_date: Optional[pd.Timestamp]
    recovery_25_time: Optional[int]

    recovery_50_date: Optional[pd.Timestamp]
    recovery_50_time: Optional[int]

    recovery_70_date: Optional[pd.Timestamp]
    recovery_70_time: Optional[int]

    recovery_90_date: Optional[pd.Timestamp]
    recovery_90_time: Optional[int]

    recovery_100_date: Optional[pd.Timestamp]
    recovery_100_time: Optional[int]