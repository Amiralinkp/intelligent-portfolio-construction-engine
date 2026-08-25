from intelligent_portfolio_construction_engine.models.feature_set import FeatureSet
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import AverageTrueRange
from ta.trend import MACD
from intelligent_portfolio_construction_engine.config.setting import Settings
import numpy as np 


class FeatureEngine:

    def __init__(self, settings):

        self.rsi_window = settings.RSI_WINDOW
        self.roc_window = settings.ROC_WINDOW
        self.atr_window = settings.ATR_WINDOW

        self.macd_fast = settings.MACD_FAST
        self.macd_slow = settings.MACD_SLOW
        self.macd_signal = settings.MACD_SIGNAL

        self.sma_short_window = settings.SMA_SHORT_WINDOW
        self.sma_long_window = settings.SMA_LONG_WINDOW
        self.dollar_volume_window = settings.DOLLAR_VOLUME_WINDOW
        




    def extract_features(self, asset_df):

        daily_ret = asset_df["Close"].pct_change()
        daily_ret = daily_ret.dropna()
        daily_return = self._daily_return(daily_ret)

        annual_return = self._annual_return(daily_ret)
        cagr = self._cagr(asset_df)
        volatility = self._volatility(daily_ret)

        sharpe_ratio = self._sharpe_ratio(daily_ret)
        sortino_ratio = self._sortino_ratio(daily_ret)

        max_drawdown = self._max_drawdown(asset_df)
        roc = self._roc(asset_df)
        rsi = self._rsi(asset_df)
        atr = self._atr(asset_df)
        current_drawdown = self._current_drawdown(asset_df)

        sma_50 = self._sma(asset_df, self.sma_short_window)
        sma_200 = self._sma(asset_df, self.sma_long_window)

        price = asset_df["Close"].iloc[-1]

        price_vs_sma_50 = (price / sma_50) - 1
        price_vs_sma_200 = (price / sma_200) - 1

        average_dollar_volume = self._average_dollar_volume(asset_df)

        macd, macd_signal, macd_hist = self._macd_features(asset_df)
        return FeatureSet(
            daily_return=daily_return,
            annual_return=annual_return,
            volatility=volatility,
            max_drawdown=max_drawdown,
            roc=roc,
            rsi=rsi,
            atr = atr,
            macd = macd,
            macd_signal = macd_signal,
            macd_hist = macd_hist,
            cagr=cagr,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            current_drawdown=current_drawdown,
            sma_50=sma_50,
            sma_200=sma_200,
            price_vs_sma_50=price_vs_sma_50,
            price_vs_sma_200=price_vs_sma_200,
            average_dollar_volume=average_dollar_volume)
    
    def _daily_return(self, daily_ret):


        return daily_ret.mean()
    
    def _annual_return(self, daily_ret):

        return (1 + daily_ret).prod() ** (252 / len(daily_ret)) - 1
    
    def _max_drawdown(self, asset_df):

        peak = asset_df["Close"].cummax()
        now_price = asset_df["Close"]
        change = ((peak - now_price)/peak)

        return change.max()
    
    def _volatility(self, daily_ret):

        volatility = daily_ret.std(ddof = 0)

        return volatility
    
    def _rsi(self, asset_df):

        close = asset_df["Close"]
        indicator = RSIIndicator(window=self.rsi_window,
                           close=close)
        
        rsi_result = indicator.rsi().iloc[-1]
        
        return rsi_result
    
    def _roc(self, asset_df):

        close = asset_df["Close"]
        indicator = ROCIndicator(window=self.roc_window,
                           close=close)
        
        roc_result = indicator.roc().iloc[-1]
        
        return roc_result

    def _atr(self, asset_df):

        high = asset_df["High"]
        low = asset_df["Low"]
        close = asset_df["Close"]
        indicator = AverageTrueRange(window=self.atr_window, close=close, high=high, low=low)

        atr_result = indicator.average_true_range().iloc[-1]

        return atr_result

      
        
    def _macd(self, asset_df):

        close = asset_df["Close"]

        indicator = MACD(
            close=close,
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal)
        

        return indicator.macd().iloc[-1]

    def _macd_features(self, asset_df):

        indicator = MACD(
            close=asset_df["Close"],
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal)

        macd = indicator.macd().iloc[-1]
        signal = indicator.macd_signal().iloc[-1]
        histogram = indicator.macd_diff().iloc[-1]

        return macd, signal, histogram

    def _cagr(self, asset_df):

        start_price = asset_df["Close"].iloc[0]
        end_price = asset_df["Close"].iloc[-1]
        years = len(asset_df) / 252

        cagr = (end_price / start_price) ** (1 / years) - 1

        return cagr


    def _sharpe_ratio(self, daily_ret):

        risk_free_rate = 0.02

        annual_mean_return = daily_ret.mean() * 252
        
        annual_volatility = daily_ret.std(ddof=0) * np.sqrt(252)
        if annual_volatility==0:
                    return 0 

        sharpe = (annual_mean_return - risk_free_rate) / annual_volatility

        return sharpe


    def _sortino_ratio(self, daily_ret):

        risk_free_rate = 0.02

        annual_mean_return = daily_ret.mean() * 252

        daily_rf = risk_free_rate / 252
        downside = daily_ret[daily_ret < daily_rf] 
        if downside.empty:
            return 0 
              
        downside_std = downside.std(ddof=0) * np.sqrt(252)
        if downside_std == 0:
            return 0

        sortino = (annual_mean_return - risk_free_rate) / downside_std

        return sortino

    def _current_drawdown(self, asset_df):

        close = asset_df["Close"]
        current_price = close.iloc[-1]
        current_peak = close.cummax().iloc[-1]

        return (current_peak - current_price) / current_peak

    def _sma(self, asset_df, window):

        return asset_df["Close"].rolling(window=window).mean().iloc[-1]


    def _average_dollar_volume(self, asset_df):

        dollar_volume = asset_df["Close"] * asset_df["Volume"]
        return dollar_volume.tail(self.dollar_volume_window).mean()