
import pandas as pd
from intelligent_portfolio_construction_engine.features.behavioral_profiling import CLUSTERING_FEATURES
from intelligent_portfolio_construction_engine.models.historical_feature_series import HistoricalFeatureSeries

def build_behavioral_snapshots(historical_series: HistoricalFeatureSeries, symbol: str, asset_class: str,frequency: str = "W"):
    feature_series = {
        "volatility": historical_series.volatility,
        "current_drawdown": historical_series.drawdown,
        "cagr": historical_series.cagr,
        "momentum_factor": historical_series.momentum_factor,
        "beta": historical_series.beta,
        "sharpe_ratio": historical_series.sharpe_ratio,
        "sortino_ratio": historical_series.sortino_ratio,}

    features = pd.DataFrame(feature_series).sort_index()

    snapshots = features.resample(frequency).last()
    snapshots = snapshots.reindex(columns=CLUSTERING_FEATURES)
    snapshots["symbol"] = symbol
    snapshots["asset_class"] = asset_class

    return snapshots.dropna(subset=CLUSTERING_FEATURES)
