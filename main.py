from intelligent_portfolio_construction_engine.data.providers.yahoo import YahooProvider
from intelligent_portfolio_construction_engine.data.processing.splitter import DataSplitter
from intelligent_portfolio_construction_engine.config.setting import Settings
from intelligent_portfolio_construction_engine.features.featur_engin import FeatureEngine
from intelligent_portfolio_construction_engine.analysis.asset_analyzer import AssetAnalyzer
from intelligent_portfolio_construction_engine.scoring.feature_scorer import FeatureScorer


provider = YahooProvider()
splitter = DataSplitter()

symbols = [
    "AAPL",
    "MSFT",
    "GOOG",
    "AMZN",
    "NVDA",
]


data = provider.get_data(
    symbols=symbols,
    start="2020-01-01",
    end="2025-01-01",
)

assets = splitter.split_assets(data)

settings = Settings()

feature_engine = FeatureEngine(settings)

analyzer = AssetAnalyzer(
    feature_engine=feature_engine,
    provider=provider)


profiles = []

for symbol, asset_df in assets.items():

    profile = analyzer.analyze(symbol, asset_df)

    profiles.append(profile)


weights = [
    0.15,  # daily_return
    0.35,  # annual_return
    0.20,  # volatility
    0.15,  # max_drawdown
    0.10,  # roc
    0.05  # atr
]


scorer = FeatureScorer(
    profiles=profiles,
    weights=weights,
)

ranked_profiles = scorer.score()


for profile in ranked_profiles:
    print(profile)