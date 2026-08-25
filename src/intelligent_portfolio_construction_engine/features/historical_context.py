from intelligent_portfolio_construction_engine.models.feature_set import FeatureSet
from intelligent_portfolio_construction_engine.models.historical_context import HistoricalContext
import numpy as np
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from intelligent_portfolio_construction_engine.models.historical_context import HistoricalContext
from intelligent_portfolio_construction_engine.models.historical_feature_series import HistoricalFeatureSeries
from intelligent_portfolio_construction_engine.models.feature_set import FeatureSet
from intelligent_portfolio_construction_engine.utils.statistics import calculate_percentile

class HistoricalContextEngine:

    def __init__(self, settings):
        self.lookback_years = settings.HISTORICAL_LOOKBACK_YEARS

    def analyze(self, asset_df, features: FeatureSet):

        historical_series = self.build_feature_series(asset_df)
        historical_context = self.calculate_percentiles(historical_series, features)

        return historical_context

    def build_feature_series(self, asset_df):

        close = asset_df["Close"]
        high = asset_df["High"]
        low = asset_df["Low"]
        volume = asset_df["Volume"]

        daily_ret = close.pct_change()

        rsi = RSIIndicator(
            close=close,
            window=14).rsi()

        roc = ROCIndicator(
            close=close,
            window=12).roc()

        atr = AverageTrueRange(
            high=high,
            low=low,
            close=close,
            window=14).average_true_range()

        macd_indicator = MACD(
            close=close,
            window_fast=12,
            window_slow=26,
            window_sign=9)

        macd = macd_indicator.macd()
        macd_signal = macd_indicator.macd_signal()
        macd_hist = macd_indicator.macd_diff()

        volatility = daily_ret.rolling(252).std(ddof=0)

        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()

        price_vs_sma_50 = (close / sma_50) - 1
        price_vs_sma_200 = (close / sma_200) - 1

        peak = close.cummax()

        drawdown = (peak - close) / peak

        dollar_volume = close * volume

        average_dollar_volume = dollar_volume.rolling(
            self.dollar_volume_window).mean()
        
        returns = asset_df["Close"].pct_change().dropna()
        rolling_sharpe = self._rolling_sharpe(returns)

        return HistoricalFeatureSeries(
            returns=returns,
            rsi=rsi,
            roc=roc,
            volatility=volatility,
            atr=atr,
            drawdown=drawdown,
            sma_50=sma_50,
            sma_200=sma_200,
            price_vs_sma_50=price_vs_sma_50,
            price_vs_sma_200=price_vs_sma_200,
            macd=macd,
            macd_signal=macd_signal,
            macd_hist=macd_hist,
            sharpe_ratio=rolling_sharpe,
            average_dollar_volume=average_dollar_volume)

    def calculate_percentiles(self, historical_series, features):

        rsi_percentile = calculate_percentile(historical_series.rsi, features.rsi)
        roc_percentile = calculate_percentile(historical_series.roc,features.roc)
        volatility_percentile = calculate_percentile(historical_series.volatility, features.volatility)
        atr_percentile = calculate_percentile(historical_series.atr, features.atr)
        price_vs_sma_50_percentile = calculate_percentile(historical_series.price_vs_sma_50, features.price_vs_sma_50)
        price_vs_sma_200_percentile = calculate_percentile(historical_series.price_vs_sma_200, features.price_vs_sma_200)
        sortino_percentile = calculate_percentile(historical_series.sortino_ratio, features.sortino_ratio)
        sharpe_percentile = calculate_percentile(historical_series.sharpe_ratio, features.sharpe_ratio)

        return HistoricalContext(
            rsi_percentile=rsi_percentile,
            roc_percentile=roc_percentile,
            volatility_percentile=volatility_percentile,
            atr_percentile=atr_percentile,
            price_vs_sma_50_percentile=price_vs_sma_50_percentile,
            price_vs_sma_200_percentile=price_vs_sma_200_percentile,
            sortino_percentile=sortino_percentile,
            sharpe_percentile=sharpe_percentile)

    def rolling_sharpe(self, returns):

        window = 252
        risk_free_rate = 0.02

        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std(ddof=0)

        annualized_return = rolling_mean*252
        annualized_volatility = rolling_std * np.sqrt(252)

        rolling_sharpe = (annualized_return - risk_free_rate) / annualized_volatility

        return rolling_sharpe