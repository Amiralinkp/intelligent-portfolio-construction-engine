from dataclasses import dataclass
from intelligent_portfolio_construction_engine.models.investment_horizon import InvestmentHorizon, PortfolioObjective

@dataclass(frozen=True)
class FeatureWeightConfig:
    sharpe_ratio: float
    sortino_ratio: float
    cagr: float
    annual_return: float
    volatility: float
    current_drawdown: float
    horizon_recovery_time: float
    momentum_factor: float
    beta: float
    average_dollar_volume: float


@dataclass(frozen=True)
class FeatureLookbackConfig:
    sharpe_ratio: int
    sortino_ratio: int
    cagr: int
    annual_return: int
    volatility: int
    momentum_factor: int
    beta: int


@dataclass(frozen=True)
class PortfolioPolicy:
    objective: PortfolioObjective
    horizon: InvestmentHorizon


@dataclass(frozen=True)
class PolicyConfig:
    policy: PortfolioPolicy
    weights: FeatureWeightConfig
    lookbacks: FeatureLookbackConfig


POLICY_CONFIGS: dict[tuple[PortfolioObjective, InvestmentHorizon], PolicyConfig] = {}

def _create_policy_config(objective, horizon, weights, lookbacks):
    return PolicyConfig(policy=PortfolioPolicy(objective=objective,horizon=horizon), weights=weights, lookbacks=lookbacks,)

def get_policy_config(objective, horizon):
    return POLICY_CONFIGS[(objective, horizon)]