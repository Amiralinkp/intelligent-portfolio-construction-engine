from dataclasses import dataclass




@dataclass
class DrawdownStatistics:

    drawdown_frequency: int

    average_drawdown: float

    max_drawdown: float

    median_recovery_25: float | None
    median_recovery_50: float | None
    median_recovery_70: float | None
    median_recovery_90: float | None
    median_recovery_100: float | None

    weighted_recovery_25: float | None
    weighted_recovery_50: float | None
    weighted_recovery_70: float | None
    weighted_recovery_90: float | None
    weighted_recovery_100: float | None
    recovery_success_rate: float
    unrecovered_count: int