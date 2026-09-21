from dataclasses import dataclass


@dataclass(frozen=True)
class BehavioralProfile:

    symbol: str
    asset_class: str
    volatility: float
    current_drawdown: float
    cagr: float
    momentum_factor: float
    beta: float
    sharpe_ratio: float
    sortino_ratio: float