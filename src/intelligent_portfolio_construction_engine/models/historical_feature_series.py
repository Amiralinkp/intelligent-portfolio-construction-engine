from dataclasses import dataclass
import pandas as pd



@dataclass
class HistoricalFeatureSeries:

    rsi: pd.Series
    roc: pd.Series
    returns: pd.Series
    volatility: pd.Series
    atr: pd.Series
    drawdown: pd.Series
    sharpe_ratio: pd.Series
    sma_50: pd.Series
    sma_200: pd.Series

    price_vs_sma_50: pd.Series
    price_vs_sma_200: pd.Series

    macd: pd.Series
    macd_signal: pd.Series
    macd_hist: pd.Series

    average_dollar_volume: pd.Series