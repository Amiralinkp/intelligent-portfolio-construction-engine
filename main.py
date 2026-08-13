from intelligent_portfolio_construction_engine.data.providers.yahoo import YahooProvider
from intelligent_portfolio_construction_engine.data.processing.splitter import DataSplitter
from intelligent_portfolio_construction_engine.config.setting import Settings
from intelligent_portfolio_construction_engine.features.featur_engin import FeatureEngine
from intelligent_portfolio_construction_engine.analysis.asset_analyzer import AssetAnalyzer
from intelligent_portfolio_construction_engine.scoring.feature_scorer import FeatureScorer
from intelligent_portfolio_construction_engine.models.portfolio_config import PortfolioConfig, PortfolioObjective
from intelligent_portfolio_construction_engine.portfolio.selection_policy import PortfolioSelectionPolicy
from intelligent_portfolio_construction_engine.portfolio.portfolio_builder import PortfolioBuilder


provider = YahooProvider()
splitter = DataSplitter()

symbols = [
    "AAPL",
    "MSFT",
    "GOOG",
    "AMZN",
    "NVDA",
    "META"
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

config = PortfolioConfig(
    capital=2500,
    objective=PortfolioObjective.RETURN_FOCUSED)

scorer = FeatureScorer(
    profiles=profiles,
    config=config)

ranked_profiles = scorer.score()

builder = PortfolioBuilder(
    profiles=ranked_profiles,
    config=config,)

assets, weights, shares, cash, metrics = builder.build()

print("\nShares")

for symbol, quantity in shares.items():
    print(f"{symbol}: {quantity}")

print("\nSelected Assets")


for asset in assets:
    print(asset.symbol)


print("\nWeights")
for symbol, weight in weights.items():
    print(f"{symbol}: {weight:.2%}")


print("\nMetrics")
print(f"Expected Return : {metrics.expected_return:.2%}")
print(f"Volatility      : {metrics.volatility:.2%}")
print(f"Sharpe Ratio    : {metrics.sharpe_ratio:.2f}")
print(f"\nRemaining Cash: ${cash:.2f}")



for profile in profiles:

    f = profile.features

    print(f"\n{profile.symbol}")

    print(f"Annual Return : {f.annual_return:.2%}")
    print(f"CAGR          : {f.cagr:.2%}")
    print(f"Volatility    : {f.volatility:.2%}")
    print(f"Max Drawdown  : {f.max_drawdown:.2%}")
    print(f"Sharpe        : {f.sharpe_ratio:.2f}")
    print(f"Sortino       : {f.sortino_ratio:.2f}")
    print(f"ROC           : {f.roc:.2f}")
    print(f"RSI           : {f.rsi:.2f}")
    print(f"ATR           : {f.atr:.2f}")