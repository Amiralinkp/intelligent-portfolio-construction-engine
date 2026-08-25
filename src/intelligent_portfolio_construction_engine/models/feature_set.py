from dataclasses import dataclass




@dataclass
class FeatureSet:

    # Performance
    daily_return: float
    annual_return: float
    cagr: float

    # Risk
    volatility: float
    atr: float
    max_drawdown: float
    current_drawdown: float

    # Momentum
    rsi: float
    roc: float

    # Trend
    sma_50: float
    sma_200: float
    price_vs_sma_50: float
    price_vs_sma_200: float
    macd: float
    macd_signal: float
    macd_hist: float

    # Liquidity
    average_dollar_volume: float

    # Risk-adjusted performance
    sharpe_ratio: float
    sortino_ratio: float