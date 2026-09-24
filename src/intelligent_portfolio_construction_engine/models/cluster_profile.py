from dataclasses import dataclass

@dataclass(frozen=True)
class ClusterProfile:
    cluster_id: int
    asset_count: int
    volatility: float
    current_drawdown: float
    cagr: float
    momentum_factor: float
    beta: float
    sharpe_ratio: float
    sortino_ratio: float