from dataclasses import dataclass



@dataclass
class PortfolioMetrics:

    expected_return: float
    volatility: float
    sharpe_ratio: float