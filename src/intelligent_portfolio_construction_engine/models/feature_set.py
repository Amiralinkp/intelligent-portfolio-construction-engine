from dataclasses import dataclass




@dataclass
class FeatureSet:

    daily_return : float

    annual_return : float

    volatility : float

    max_drawdown : float

    rsi : float

    roc : float

    atr : float

    macd :float

    macd_signal : float

    macd_hist : float