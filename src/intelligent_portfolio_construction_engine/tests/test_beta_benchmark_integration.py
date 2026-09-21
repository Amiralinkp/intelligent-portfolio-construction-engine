import numpy as np
import pandas as pd

from intelligent_portfolio_construction_engine.config.benchmarks import (
    get_benchmark,
)
from intelligent_portfolio_construction_engine.features.featur_engin import (
    FeatureEngine,
)
from intelligent_portfolio_construction_engine.features.historical_context import (
    HistoricalContextEngine,
)


class TestSettings:
    RSI_WINDOW = 14
    ROC_WINDOW = 20
    ATR_WINDOW = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    SMA_SHORT_WINDOW = 50
    SMA_LONG_WINDOW = 200
    DOLLAR_VOLUME_WINDOW = 20
    BETA_WINDOW = 20
    MOMENTUM_WINDOW = 20
    HISTORICAL_LOOKBACK_YEARS = 6


class MockProvider:
    def __init__(self, benchmark_df):
        self.benchmark_df = benchmark_df
        self.requested_asset_class = None

    def get_data(self, symbols, start, end):
        self.requested_asset_class = symbols[0]

        return self.benchmark_df


def build_market_data(size=300):
    dates = pd.date_range(
        start="2024-01-01",
        periods=size,
        freq="B",
    )

    rng = np.random.default_rng(42)

    benchmark_returns = rng.normal(
        loc=0.001,
        scale=0.01,
        size=size,
    )

    asset_returns = (
        1.5 * benchmark_returns
        + rng.normal(
            loc=0.0,
            scale=0.001,
            size=size,
        )
    )

    benchmark_close = 100 * np.cumprod(1 + benchmark_returns)
    asset_close = 100 * np.cumprod(1 + asset_returns)

    benchmark_df = pd.DataFrame(
        {
            "Close": benchmark_close,
        },
        index=dates,
    )

    asset_df = pd.DataFrame(
        {
            "Open": asset_close,
            "High": asset_close * 1.01,
            "Low": asset_close * 0.99,
            "Close": asset_close,
            "Volume": np.full(size, 100_000),
        },
        index=dates,
    )

    return asset_df, benchmark_df


def test_benchmark_configuration():
    equity_benchmark = get_benchmark("equity")
    gold_benchmark = get_benchmark("gold")
    crypto_benchmark = get_benchmark("crypto")

    assert equity_benchmark.symbol == "^GSPC"
    assert gold_benchmark.symbol == "GC=F"
    assert crypto_benchmark.symbol == "BTC-USD"


def test_feature_engineering_calculates_beta_from_benchmark():
    settings = TestSettings()
    feature_engineering = FeatureEngine(settings)

    asset_df, benchmark_df = build_market_data()

    features = feature_engineering.extract_features(
        asset_df,
        benchmark_df,
    )

    assert features.beta is not None
    assert np.isfinite(features.beta)

    assert np.isclose(
        features.beta,
        1.5,
        atol=0.15,
    )


def test_historical_context_calculates_beta_from_benchmark():
    settings = TestSettings()

    asset_df, benchmark_df = build_market_data()

    provider = MockProvider(benchmark_df)

    engine = HistoricalContextEngine(
        settings,
        provider,
    )

    benchmark_symbol = get_benchmark("equity").symbol

    benchmark_data = engine._get_benchmark_data(
        asset_class="equity",
        start_date=asset_df.index.min().strftime("%Y-%m-%d"),
        end_date=asset_df.index.max().strftime("%Y-%m-%d"),
    )

    assert provider.requested_asset_class == benchmark_symbol
    assert benchmark_data.equals(benchmark_df)

    daily_asset_returns = asset_df["Close"].pct_change()
    daily_benchmark_returns = benchmark_df["Close"].pct_change()

    beta_series = engine.calculate_rolling_beta(
        daily_asset_returns,
        daily_benchmark_returns,
    )

    beta = beta_series.dropna().iloc[-1]

    assert np.isfinite(beta)
    assert np.isclose(
        beta,
        1.5,
        atol=0.15,
    )


def test_historical_context_receives_the_correct_benchmark():
    settings = TestSettings()

    asset_df, benchmark_df = build_market_data()

    provider = MockProvider(benchmark_df)

    engine = HistoricalContextEngine(
        settings,
        provider,
    )

    engine._get_benchmark_data(
        asset_class="equity",
        start_date=asset_df.index.min().strftime("%Y-%m-%d"),
        end_date=asset_df.index.max().strftime("%Y-%m-%d"),
    )

    expected_symbol = get_benchmark("equity").symbol

    assert provider.requested_asset_class == expected_symbol