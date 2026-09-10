import numpy as np
import pandas as pd

from intelligent_portfolio_construction_engine.features.historical_context import HistoricalContextEngine
from intelligent_portfolio_construction_engine.models.feature_set import FeatureSet
from intelligent_portfolio_construction_engine.models.historical_feature_series import HistoricalFeatureSeries
from intelligent_portfolio_construction_engine.models.historical_analysis import HistoricalAnalysis


class TestSettings:
    HISTORICAL_LOOKBACK_YEARS = 2
    DOLLAR_VOLUME_WINDOW = 20


def create_engine():
    return HistoricalContextEngine(TestSettings())


def create_asset_data(start="2021-01-01", end="2024-12-31"):
    dates = pd.date_range(start, end, freq="D")

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
            "Volume": np.full(len(dates), 1000),
        },
        index=dates,
    )


def create_historical_series():
    dates = pd.date_range("2021-01-01", "2024-12-31", freq="D")

    returns = pd.Series(0.001, index=dates)

    return HistoricalFeatureSeries(
        rsi=pd.Series(50.0, index=dates),
        roc=pd.Series(1.0, index=dates),
        returns=returns,
        volatility=pd.Series(0.2, index=dates),
        atr=pd.Series(1.0, index=dates),
        drawdown=pd.Series(0.1, index=dates),
        sharpe_ratio=pd.Series(1.0, index=dates),
        sortino_ratio=pd.Series(1.5, index=dates),
        sma_50=pd.Series(100.0, index=dates),
        sma_200=pd.Series(100.0, index=dates),
        price_vs_sma_50=pd.Series(0.01, index=dates),
        price_vs_sma_200=pd.Series(0.02, index=dates),
        macd=pd.Series(1.0, index=dates),
        macd_signal=pd.Series(0.8, index=dates),
        macd_hist=pd.Series(0.2, index=dates),
        average_dollar_volume=pd.Series(100000.0, index=dates),
    )


def create_feature_set():
    return FeatureSet(
        daily_return=0.001,
        annual_return=0.25,
        cagr=0.20,
        volatility=0.20,
        atr=1.0,
        max_drawdown=0.10,
        current_drawdown=0.05,
        rsi=50.0,
        roc=1.0,
        sma_50=100.0,
        sma_200=100.0,
        price_vs_sma_50=0.01,
        price_vs_sma_200=0.02,
        macd=1.0,
        macd_signal=0.8,
        macd_hist=0.2,
        average_dollar_volume=100000.0,
        sharpe_ratio=1.0,
        sortino_ratio=1.5,
    )


def test_build_feature_series():
    engine = create_engine()
    asset_df = create_asset_data()

    result = engine.build_feature_series(asset_df)

    assert isinstance(result, HistoricalFeatureSeries)

    assert len(result.returns) == len(asset_df) - 1
    assert len(result.rsi) == len(asset_df)
    assert len(result.roc) == len(asset_df)
    assert len(result.volatility) == len(asset_df)
    assert len(result.atr) == len(asset_df)

    assert len(result.sma_50) == len(asset_df)
    assert len(result.sma_200) == len(asset_df)

    assert len(result.drawdown) == len(asset_df)
    assert len(result.average_dollar_volume) == len(asset_df)

    assert result.sharpe_ratio.notna().any()
    assert result.sortino_ratio.notna().any()


def test_build_seasonal_behavior():
    engine = create_engine()
    historical_series = create_historical_series()

    result = engine.build_seasonal_behavior(historical_series)

    assert len(result) == 4

    seasons = [behavior.season for behavior in result]

    assert seasons == [1, 2, 3, 4]

    q1 = result[0]

    expected_return = (1 + 0.001) ** 90 - 1

    assert np.isclose(q1.average_return, expected_return)
    assert np.isclose(q1.average_volatility, 0.2)
    assert np.isclose(q1.average_sharpe, engine.calculate_sharpe(
        pd.Series(0.001, index=historical_series.returns.index[:90])
    ))
    assert np.isclose(q1.average_sortino, 0.0)

    assert np.isclose(q1.average_max_drawdown, 0.1)
    assert np.isclose(q1.average_positive_return_rate, 100.0)
    assert np.isclose(q1.average_liquidity, 100000.0)
    assert np.isclose(q1.average_distance_from_peak, 0.1)


def test_build_seasonal_context():
    engine = create_engine()
    historical_series = create_historical_series()

    result = engine.build_seasonal_context(historical_series)

    assert len(result.seasonal_behavior) == 4

    assert len(result.sharpe_comparisons) == 8
    assert len(result.sortino_comparisons) == 8
    assert len(result.liquidity_comparisons) == 8
    assert len(result.drawdown_comparisons) == 8
    assert len(result.positive_return_comparisons) == 8

    first_sharpe = result.sharpe_comparisons[0]

    assert first_sharpe.season in {1, 2, 3, 4}
    assert first_sharpe.current_year == 2024
    assert first_sharpe.historical_year in {2022, 2023}

    assert np.isclose(
        first_sharpe.difference,
        first_sharpe.current_value - first_sharpe.historical_value,
    )

    first_liquidity = result.liquidity_comparisons[0]

    assert np.isclose(first_liquidity.current_value, 100000.0)
    assert np.isclose(first_liquidity.historical_value, 100000.0)
    assert np.isclose(first_liquidity.difference, 0.0)


def test_calculate_percentiles():
    engine = create_engine()
    historical_series = create_historical_series()
    features = create_feature_set()

    seasonal_context = engine.build_seasonal_context(historical_series)

    result = engine.calculate_percentiles(
        historical_series,
        features,
        seasonal_context,
    )

    assert result.seasonal_context is seasonal_context

    assert np.isclose(result.rsi_percentile, 1.0)
    assert np.isclose(result.roc_percentile, 1.0)
    assert np.isclose(result.volatility_percentile, 1.0)
    assert np.isclose(result.atr_percentile, 1.0)

    assert np.isclose(result.price_vs_sma_50_percentile, 1.0)
    assert np.isclose(result.price_vs_sma_200_percentile, 1.0)

    assert np.isclose(result.sharpe_percentile, 1.0)
    assert np.isclose(result.sortino_percentile, 1.0)


def test_analyze():
    engine = create_engine()
    asset_df = create_asset_data()
    features = create_feature_set()

    result = engine.analyze(asset_df, features)

    assert isinstance(result, HistoricalAnalysis)

    assert result.historical_context is not None
    assert result.drawdown_statistics is not None

    historical_context = result.historical_context
    drawdown_statistics = result.drawdown_statistics

    assert historical_context.seasonal_context is not None

    assert len(
        historical_context.seasonal_context.seasonal_behavior
    ) == 4

    assert len(
        historical_context.seasonal_context.sharpe_comparisons
    ) == 8

    assert len(
        historical_context.seasonal_context.sortino_comparisons
    ) == 8

    assert len(
        historical_context.seasonal_context.liquidity_comparisons
    ) == 8

    assert len(
        historical_context.seasonal_context.drawdown_comparisons
    ) == 8

    assert len(
        historical_context.seasonal_context.positive_return_comparisons
    ) == 8

    assert isinstance(drawdown_statistics.max_drawdown, (int, float))
    assert drawdown_statistics.max_drawdown >= 0