from intelligent_portfolio_construction_engine.data.providers.yahoo import YahooProvider
from intelligent_portfolio_construction_engine.data.processing.splitter import DataSplitter
from intelligent_portfolio_construction_engine.config.setting import Settings
from intelligent_portfolio_construction_engine.features.featur_engin import FeatureEngine
from intelligent_portfolio_construction_engine.analysis.asset_analyzer import AssetAnalyzer



provider = YahooProvider()
splitter  = DataSplitter()
symbols = [
    "AAPL",
    "MSFT",
    "GOOG",
    "AMZN",
    "NVDA"]


data = provider.get_data(
    symbols = symbols,
    start="2020-01-01",
    end="2025-01-01")

assets = splitter.split_assets(data)
settings = Settings()

feature_engine = FeatureEngine(settings)

analyzer = AssetAnalyzer(feature_engine)

for symbol, asset_df in assets.items():

    profile = analyzer.analyze(symbol, asset_df)

print(profile)

