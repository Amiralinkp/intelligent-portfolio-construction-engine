from dataclasses import dataclass
from intelligent_portfolio_construction_engine.models.feature_set import FeatureSet


@dataclass
class AssetProfile:

    symbol: str

    features : FeatureSet

    score: float = 0.0

    rank: int | None = None

    recommendation : str | None = None