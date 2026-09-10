from dataclasses import dataclass
from .seasonal_context import SeasonalContext

@dataclass
class HistoricalContext:

    # Momentum
    rsi_percentile: float
    roc_percentile: float

    # Risk
    volatility_percentile: float
    atr_percentile: float

    # Trend
    price_vs_sma_50_percentile: float
    price_vs_sma_200_percentile: float

    # Performance
    sharpe_percentile: float
    sortino_percentile: float

    # Seasonal Historical Context
    seasonal_context: SeasonalContext