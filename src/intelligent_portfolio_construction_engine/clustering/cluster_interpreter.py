from intelligent_portfolio_construction_engine.models.cluster_interpretation import ClusterInterpretation
from intelligent_portfolio_construction_engine.models.cluster_profile import ClusterProfile




def _relative_level(value, values):

    if len(values) == 1:
        return "medium"

    sorted_values = sorted(values)
    rank = sorted_values.index(value)

    if rank == 0:
        return "low"

    if rank == len(sorted_values) - 1:
        return "high"

    return "medium"

    
def interpret_cluster_profiles(profiles):

    volatility_values = [profile.volatility for profile in profiles]

    current_drawdown_values = [profile.current_drawdown for profile in profiles]

    cagr_values = [profile.cagr for profile in profiles]

    momentum_values = [profile.momentum_factor for profile in profiles]

    beta_values = [profile.beta for profile in profiles]

    sharpe_values = [profile.sharpe_ratio for profile in profiles]

    sortino_values = [profile.sortino_ratio for profile in profiles]


    interpretations = []

    for profile in profiles:
            
        interpretation = ClusterInterpretation(
            cluster_id=profile.cluster_id,
            volatility_level=_relative_level(profile.volatility, volatility_values),
            drawdown_level=_relative_level(profile.current_drawdown, current_drawdown_values),
            beta_level=_relative_level(profile.beta, beta_values),
            return_profile=_relative_level(profile.cagr, cagr_values),
            momentum_profile=_relative_level(profile.momentum_factor, momentum_values),
            sharpe_level=_relative_level(profile.sharpe_ratio, sharpe_values),
            sortino_level=_relative_level(profile.sortino_ratio, sortino_values),)

        interpretations.append(interpretation)

    return interpretations
