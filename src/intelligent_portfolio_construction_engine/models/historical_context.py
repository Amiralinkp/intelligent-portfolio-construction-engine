
from dataclasses import dataclass



@dataclass
class HistoricalContext:

    # Momentum
    rsi_percentile: float
    roc_percentile: float

    # Risk
    volatility_percentile: float
    drawdown_percentile: float
    atr_percentile: float

    # Trend
    price_vs_sma_50_percentile: float
    price_vs_sma_200_percentile: float

    # Performance
    return_percentile: float
    sharpe_percentile: float
    sortino_percentile: float