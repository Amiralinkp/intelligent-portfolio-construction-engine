from intelligent_portfolio_construction_engine.models.cluster_interpretation import ClusterInterpretation


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
    feature_names = {
        "volatility": "volatility_level",
        "current_drawdown": "drawdown_level",
        "cagr": "return_profile",
        "momentum_factor": "momentum_profile",
        "beta": "beta_level",
        "sharpe_ratio": "sharpe_level",
        "sortino_ratio": "sortino_level",
    }

    feature_values = {
        feature: [
            profile.median_profile[feature]
            for profile in profiles
        ]
        for feature in feature_names
    }

    interpretations = []

    for profile in profiles:
        levels = {
            output_name: _relative_level(
                profile.median_profile[feature],
                feature_values[feature],
            )
            for feature, output_name in feature_names.items()
        }

        interpretation = ClusterInterpretation(
            cluster_id=profile.cluster_id,
            **levels,
        )

        interpretations.append(interpretation)

    return interpretations