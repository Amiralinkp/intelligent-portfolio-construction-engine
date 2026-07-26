from intelligent_portfolio_construction_engine.models.feature_set import FeatureSet
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import AverageTrueRange
from ta.trend import MACD
from intelligent_portfolio_construction_engine.config.setting import Settings


class FeatureEngine:

    def __init__(self, settings):

        self.rsi_window = settings.RSI_WINDOW
        self.roc_window = settings.ROC_WINDOW
        self.atr_window = settings.ATR_WINDOW

        self.macd_fast = settings.MACD_FAST
        self.macd_slow = settings.MACD_SLOW
        self.macd_signal = settings.MACD_SIGNAL




    def extract_features(self, asset_df):

        daily_ret = asset_df["Close"].pct_change()
        daily_ret = daily_ret.dropna()
        daily_return = self._daily_return(daily_ret)

        annual_return = self._annual_return(daily_ret)
        volatility = self._volatility(daily_ret)

        max_drawdown = self._max_drawdown(asset_df)
        roc = self._roc(asset_df)
        rsi = self._rsi(asset_df)
        atr = self._atr(asset_df)
        macd = self._macd(asset_df)
        macd_signal = self._macd_signal(asset_df)
        macd_hist = self._macd_hist(asset_df)

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
            macd_hist = macd_hist)
    
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

    def _macd_signal(self, asset_df):
        close = asset_df["Close"]

        indicator = MACD(
            close=close,
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal)

        return indicator.macd_signal().iloc[-1]

    def _macd_hist(self, asset_df):

        close = asset_df["Close"]

        indicator = MACD(
            close=close,
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal)
        
        return indicator.macd_diff().iloc[-1]