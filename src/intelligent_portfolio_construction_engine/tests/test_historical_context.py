import numpy as np
import pandas as pd
import pytest

from intelligent_portfolio_construction_engine.features.historical_context import (
    HistoricalContextEngine,
)
from intelligent_portfolio_construction_engine.models.feature_set import FeatureSet
from intelligent_portfolio_construction_engine.models.historical_analysis import (
    HistoricalAnalysis,
)
from intelligent_portfolio_construction_engine.models.historical_feature_series import (
    HistoricalFeatureSeries,
)


class TestSettings:
    HISTORICAL_LOOKBACK_YEARS = 2
    DOLLAR_VOLUME_WINDOW = 20
    BETA_WINDOW = 252
    MOMENTUM_WINDOW = 126


def create_asset_data():
    dates = pd.date_range("2021-01-01", "2024-12-31", freq="D")

    returns = np.where(
        np.arange(len(dates)) % 2 == 0,
        0.001,
        -0.0005,
    )

    close = 100 * np.cumprod(1 + returns)

    return pd.DataFrame(
        {
            "Close": close,
            "High": close + 1,
            "Low": close - 1,
            "Volume": 1000,
        },
        index=dates,
    )


def create_benchmark_data():
    dates = pd.date_range("2021-01-01", "2024-12-31", freq="D")

    returns = np.where(
        np.arange(len(dates)) % 2 == 0,
        0.0005,
        -0.0005,
    )

    close = 100 * np.cumprod(1 + returns)

    return pd.DataFrame(
        {
            "Close": close,
        },
        index=dates,
    )


def create_historical_series():
    dates = pd.date_range("2021-01-01", "2024-12-31", freq="D")

    constant = lambda value: pd.Series(value, index=dates)

    return HistoricalFeatureSeries(
        rsi=constant(50),
        roc=constant(1),
        returns=constant(0.001),
        volatility=constant(0.2),
        atr=constant(1),
        drawdown=constant(0.1),
        sharpe_ratio=constant(1),
        sortino_ratio=constant(1.5),
        sma_50=constant(100),
        sma_200=constant(100),
        price_vs_sma_50=constant(0.01),
        price_vs_sma_200=constant(0.02),
        macd=constant(1),
        macd_signal=constant(0.8),
        macd_hist=constant(0.2),
        average_dollar_volume=constant(100000),
        beta=constant(1.0),
        momentum_factor=constant(0.10),
    )


def create_feature_set():
    return FeatureSet(
        daily_return=0.001,
        annual_return=0.25,
        cagr=0.20,
        volatility=0.2,
        atr=1,
        max_drawdown=0.1,
        current_drawdown=0.05,
        rsi=50,
        roc=1,
        momentum_factor=0.10,
        sma_50=100,
        sma_200=100,
        price_vs_sma_50=0.01,
        price_vs_sma_200=0.02,
        macd=1,
        macd_signal=0.8,
        macd_hist=0.2,
        average_dollar_volume=100000,
        beta=1.0,
        sharpe_ratio=1,
        sortino_ratio=1.5,
    )


@pytest.fixture
def engine():
    return HistoricalContextEngine(TestSettings())


def test_build_feature_series(engine):
    asset_df = create_asset_data()
    benchmark_df = create_benchmark_data()

    result = engine.build_feature_series(
        asset_df,
        benchmark_df,
    )

    assert isinstance(result, HistoricalFeatureSeries)

    assert len(result.rsi) == len(asset_df)
    assert len(result.roc) == len(asset_df)
    assert len(result.returns) == len(asset_df) - 1

    assert len(result.sma_50) == len(asset_df)
    assert len(result.sma_200) == len(asset_df)
    assert len(result.drawdown) == len(asset_df)
    assert len(result.average_dollar_volume) == len(asset_df)

    assert result.sharpe_ratio.notna().any()
    assert result.sortino_ratio.notna().any()

    assert result.beta.notna().any()
    assert result.momentum_factor.notna().any()


def test_calculate_rolling_beta(engine):
    rng = np.random.default_rng(42)

    dates = pd.date_range(
        "2021-01-01",
        periods=400,
        freq="D",
    )

    benchmark_returns = pd.Series(
        rng.normal(0, 0.01, len(dates)),
        index=dates,
    )

    noise = pd.Series(
        rng.normal(0, 0.0001, len(dates)),
        index=dates,
    )

    asset_returns = (
        2 * benchmark_returns
        + noise
    )

    beta = engine.calculate_rolling_beta(
        asset_returns,
        benchmark_returns,
    )

    valid_beta = beta.dropna()

    assert not valid_beta.empty

    assert valid_beta.iloc[-1] == pytest.approx(
        2.0,
        abs=0.05,
    )


def test_calculate_momentum_factor(engine):
    dates = pd.date_range(
        "2021-01-01",
        periods=200,
        freq="D",
    )

    close = pd.Series(
        np.arange(100, 300, dtype=float),
        index=dates,
    )

    momentum = engine.calculate_momentum_factor(close)

    expected = close.pct_change(
        TestSettings.MOMENTUM_WINDOW
    )

    pd.testing.assert_series_equal(
        momentum,
        expected,
    )


def test_build_seasonal_behavior(engine):
    historical_series = create_historical_series()

    result = engine.build_seasonal_behavior(
        historical_series
    )

    assert len(result) == 4

    returns = historical_series.returns

    current_year = returns.index[-1].year

    for behavior in result:
        assert behavior.season in {1, 2, 3, 4}

        expected_returns = []
        expected_sharpes = []

        for year_offset in range(
            1,
            engine.lookback_years + 1,
        ):
            historical_year = current_year - year_offset

            historical_returns = engine.get_quarter_returns(
                returns,
                historical_year,
                behavior.season,
            )

            if historical_returns.empty:
                continue

            expected_returns.append(
                (1 + historical_returns).prod() - 1
            )

            expected_sharpes.append(
                engine.calculate_sharpe(
                    historical_returns
                )
            )

        expected_average_return = np.mean(
            expected_returns
        )

        expected_average_sharpe = np.mean(
            expected_sharpes
        )

        assert behavior.average_return == pytest.approx(
            expected_average_return,
            rel=1e-2,
        )

        assert behavior.average_volatility == pytest.approx(
            0.2
        )

        assert behavior.average_sharpe == pytest.approx(
            expected_average_sharpe,
            rel=1e-2,
        )

        assert behavior.average_sortino == pytest.approx(
            0
        )

        assert behavior.average_max_drawdown == pytest.approx(
            0.1
        )

        assert behavior.average_positive_return_rate == pytest.approx(
            100
        )

        assert behavior.average_liquidity == pytest.approx(
            100000
        )

        assert behavior.average_distance_from_peak == pytest.approx(
            0.1
        )


def test_calculate_percentiles(engine):
    historical_series = create_historical_series()
    features = create_feature_set()

    seasonal_context = engine.build_seasonal_context(
        historical_series
    )

    result = engine.calculate_percentiles(
        historical_series,
        features,
        seasonal_context,
    )

    assert result.rsi_percentile == pytest.approx(1)
    assert result.roc_percentile == pytest.approx(1)

    assert result.volatility_percentile == pytest.approx(1)
    assert result.atr_percentile == pytest.approx(1)

    assert result.price_vs_sma_50_percentile == pytest.approx(1)
    assert result.price_vs_sma_200_percentile == pytest.approx(1)

    assert result.sharpe_percentile == pytest.approx(1)
    assert result.sortino_percentile == pytest.approx(1)

    assert result.seasonal_context is seasonal_context


def test_analyze(engine):
    asset_df = create_asset_data()
    benchmark_df = create_benchmark_data()
    features = create_feature_set()

    result = engine.analyze(
        asset_df,
        features,
        benchmark_df,
    )

    assert isinstance(result, HistoricalAnalysis)

    assert result.historical_context is not None
    assert result.drawdown_statistics is not None

    assert len(
        result.historical_context.seasonal_context.seasonal_behavior
    ) == 4

    seasonal_context = (
        result.historical_context.seasonal_context
    )

    assert len(
        seasonal_context.sharpe_comparisons
    ) == 8

    assert len(
        seasonal_context.sortino_comparisons
    ) == 8

    assert len(
        seasonal_context.liquidity_comparisons
    ) == 8

    assert len(
        seasonal_context.drawdown_comparisons
    ) == 8

    assert len(
        seasonal_context.positive_return_comparisons
    ) == 8

    assert isinstance(
        result.drawdown_statistics.max_drawdown,
        (int, float),
    )

    assert result.drawdown_statistics.max_drawdown >= 0