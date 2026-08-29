from intelligent_portfolio_construction_engine.models.feature_set import FeatureSet
from intelligent_portfolio_construction_engine.models.historical_context import HistoricalContext
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from intelligent_portfolio_construction_engine.models.historical_feature_series import HistoricalFeatureSeries
from intelligent_portfolio_construction_engine.utils.statistics import calculate_percentile
from intelligent_portfolio_construction_engine.models.seasonal_context import SeasonalComparison, SeasonalContext


class HistoricalContextEngine:

    def __init__(self, settings):
        self.lookback_years = settings.HISTORICAL_LOOKBACK_YEARS
        self.dollar_volume_window = settings.DOLLAR_VOLUME_WINDOW

    def analyze(self, asset_df, features: FeatureSet):

        historical_series = self.build_feature_series(asset_df)
        seasonal_context = self.build_seasonal_context(historical_series.returns)
        historical_context = self.calculate_percentiles(historical_series, features, seasonal_context)

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
        rolling_sharpe = self.rolling_sharpe(returns)
        rolling_sortino = self.rolling_sortino(returns)

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
            sortino_ratio=rolling_sortino,
            average_dollar_volume=average_dollar_volume)

    def calculate_percentiles(self, historical_series, features, seasonal_context):

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
            sharpe_percentile=sharpe_percentile,
            seasonal_context=seasonal_context)

    def rolling_sharpe(self, returns):

        window = 252
        risk_free_rate = 0.02

        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std(ddof=0)

        annualized_return = rolling_mean*252
        annualized_volatility = rolling_std * np.sqrt(252)

        rolling_sharpe = (annualized_return - risk_free_rate) / annualized_volatility

        return rolling_sharpe

    def rolling_sortino(self, returns):

        window = 252
        risk_free_rate = 0.02
        rolling_mean = returns.rolling(window).mean()

        daily_rf = risk_free_rate / 252
        downside_returns = returns.where(returns < daily_rf)
        downside_std = downside_returns.rolling(window).std(ddof=0)

        annualized_return = rolling_mean * 252
        annualized_downside_volatility = downside_std * np.sqrt(252)

        rolling_sortino = (annualized_return - risk_free_rate) / annualized_downside_volatility
        rolling_sortino = rolling_sortino.replace([np.inf, -np.inf],np.nan)

        return rolling_sortino



    def calculate_sharpe(self, returns):

        risk_free_rate = 0.02

        if returns.empty:
            return 0.0

        annual_mean_return = returns.mean() * 252
        annualized_volatility = returns.std(ddof=0) * np.sqrt(252)

        if annualized_volatility == 0:
            return 0.0

        return (annual_mean_return - risk_free_rate) / annualized_volatility

    def calculate_sortino(self, returns):

        risk_free_rate = 0.02
        if returns.empty:
            return 0.0

        annual_mean_return = returns.mean() * 252
        daily_rf = risk_free_rate / 252
        downside = returns[returns < daily_rf]

        if downside.empty:
            return 0.0

        downside_std = downside.std(ddof=0) * np.sqrt(252)

        if downside_std == 0:
            return 0.0

        return (annual_mean_return - risk_free_rate) / downside_std

    def get_quarter(self, month):

        if month <= 3:
            return 1
        elif month <= 6:
            return 2
        elif month <= 9:
            return 3
        else:
            return 4

    def get_quarter_returns(self, returns, year, quarter, end_date=None):

        quarter_months = {
            1: (1, 3),
            2: (4, 6),
            3: (7, 9),
            4: (10, 12)}

        start_month, end_month = quarter_months[quarter]

        start_date = pd.Timestamp(
            year=year,
            month=start_month,
            day=1)

        if end_date is None:
            
            if end_month == 12:
                end_date = pd.Timestamp(
                year=year + 1,
                month=1,
                day=1)
            else:
                end_date = pd.Timestamp(
                year=year,
                month=end_month + 1,
                day=1)
        else:
            end_date = pd.Timestamp(
                year=year,
                month=end_date.month,
                day=end_date.day) + pd.Timedelta(days=1)

        period_returns = returns[
            (returns.index >= start_date)
            & (returns.index < end_date)]

        return period_returns

    def build_seasonal_context(self, returns):

        sharpe_comparisons = []
        sortino_comparisons = []

        current_date = returns.index[-1]
        current_year = current_date.year

        for quarter in range(1, 5):

            current_period_returns = self.get_quarter_returns(returns, current_year, quarter)

            if current_period_returns.empty:
                continue

            current_sharpe = self.calculate_sharpe(current_period_returns)

            current_sortino = self.calculate_sortino(current_period_returns)

            for year_offset in range(1, self.lookback_years + 1):

                historical_year = current_year - year_offset
                historical_end_date = current_date.replace(year=historical_year)
                historical_returns = self.get_quarter_returns(returns, historical_year, quarter, end_date=historical_end_date)

                if historical_returns.empty:
                    continue

                historical_sharpe = self.calculate_sharpe(historical_returns)

                historical_sortino = self.calculate_sortino(historical_returns)

                sharpe_comparisons.append(SeasonalComparison(
                    season=quarter,
                    current_year=current_year,
                    historical_year=historical_year,
                    current_value=current_sharpe,
                    historical_value=historical_sharpe,
                    difference=(current_sharpe- historical_sharpe)))

                sortino_comparisons.append(SeasonalComparison(
                    season=quarter,
                    current_year=current_year,
                    historical_year=historical_year,
                    current_value=current_sortino,
                    historical_value=historical_sortino,
                    difference=(current_sortino- historical_sortino)))

        return SeasonalContext(
            sharpe_comparisons=sharpe_comparisons,
            sortino_comparisons=sortino_comparisons)