from dataclasses import dataclass
import pandas as pd
from intelligent_portfolio_construction_engine.models.feature_set import FeatureSet


@dataclass
class AssetProfile:

    symbol: str

    features : FeatureSet

    sector : str

    score : float = 0.0

    rank: int | None = None

    recommendation : str | None = None

    historical_data: pd.DataFrame