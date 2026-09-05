from dataclasses import dataclass


@dataclass
class SeasonalComparison:

    season: int
    current_year: int
    historical_year: int

    current_value: float
    historical_value: float

    difference: float


@dataclass
class SeasonalBehavior:

    season: int

    average_return: float
    average_volatility: float

    average_sharpe: float
    average_sortino: float

    average_max_drawdown: float
    average_positive_return_rate: float

    average_liquidity: float
    average_distance_from_peak: float
    
@dataclass
class SeasonalContext:

    sharpe_comparisons: list[SeasonalComparison]
    sortino_comparisons: list[SeasonalComparison]

    liquidity_comparisons: list[SeasonalComparison]
    drawdown_comparisons: list[SeasonalComparison]

    positive_return_comparisons: list[SeasonalComparison]

    seasonal_behavior: list[SeasonalBehavior]

