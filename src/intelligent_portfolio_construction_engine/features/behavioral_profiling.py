from intelligent_portfolio_construction_engine.models.behavioral_profile import BehavioralProfile
from intelligent_portfolio_construction_engine.models.feature_set import FeatureSet


CLUSTERING_FEATURES = [
    "volatility",
    "current_drawdown",
    "cagr",
    "momentum_factor",
    "beta",
    "sharpe_ratio",
    "sortino_ratio"]


def build_behavioral_profile(features: FeatureSet, symbol, asset_class):

    return BehavioralProfile(
        symbol=symbol,
        asset_class=asset_class,
        volatility=features.volatility,
        current_drawdown=features.current_drawdown,
        cagr=features.cagr,
        momentum_factor=features.momentum_factor,
        beta=features.beta,
        sharpe_ratio=features.sharpe_ratio,
        sortino_ratio=features.sortino_ratio)

