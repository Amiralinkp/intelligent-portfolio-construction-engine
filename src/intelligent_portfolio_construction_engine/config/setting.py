
from dataclasses import dataclass



@dataclass(frozen=True)


class Settings:

    RSI_WINDOW : int = 14
    ROC_WINDOW : int = 20
    ATR_WINDOW : int = 14
    MACD_FAST : int = 12
    MACD_SLOW : int = 26
    MACD_SIGNAL : int = 9