import yfinance as yf
from intelligent_portfolio_construction_engine.data.interfaces.market_data_provider import MarketDataProvider
from intelligent_portfolio_construction_engine.config.benchmarks import get_benchmark


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
    
    def get_asset_info(self, symbol):

        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            "sector": info.get("sector"),
            "industry": info.get("industry")}

    def get_benchmark_data(self, asset_class: str, start: str, end: str):
        benchmark = get_benchmark(asset_class)

        return self.get_data(symbols=[benchmark.symbol], start=start, end=end)