from dataclasses import dataclass
from intelligent_portfolio_construction_engine.models.cluster_profile import ClusterProfile




@dataclass(frozen=True)

class ClusterInterpretation:
    cluster_id: int
    volatility_level: str
    drawdown_level: str
    beta_level: str
    return_profile: str
    momentum_profile: str
    sharpe_level: str
    sortino_level: str