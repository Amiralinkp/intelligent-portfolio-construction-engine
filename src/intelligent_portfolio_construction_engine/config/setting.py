from intelligent_portfolio_construction_engine.models.investment_horizon import InvestmentHorizon
from dataclasses import dataclass



@dataclass(frozen=True)


class Settings:
    RSI_WINDOW: int = 14
    ROC_WINDOW: int = 20
    ATR_WINDOW: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9

    HISTORICAL_LOOKBACK_YEARS: int = 6
    BETA_WINDOW: int = 252
    MOMENTUM_WINDOW: int = 126

    CLUSTERING_MIN_K: int = 2
    CLUSTERING_MAX_K: int = 6

    INVESTMENT_HORIZON = InvestmentHorizon.MEDIUM

    # Historical Behavioral Intelligence
    BEHAVIORAL_HISTORY_START_DATE: str = "2018-01-01"
    BEHAVIORAL_HISTORY_END_DATE: str | None = None
    BEHAVIORAL_SNAPSHOT_FREQUENCY: str = "W"