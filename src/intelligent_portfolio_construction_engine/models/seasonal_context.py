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
class SeasonalContext:

    sharpe_comparisons: list[SeasonalComparison]
    sortino_comparisons: list[SeasonalComparison]