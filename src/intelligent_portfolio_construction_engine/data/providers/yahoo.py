import yfinance as yf
from intelligent_portfolio_construction_engine.data.interfaces.market_data_provider import MarketDataProvider


class YahooProvider(MarketDataProvider):

    def get_data(self, symbols : list[str], start : str, end : str):
        
        data = yf.download(
            tickers=symbols,
            start=start,
            end=end,
            progress=False)
        
        if data.empty:
            raise ValueError(f"No market data found for : {symbols}")
        
        return data