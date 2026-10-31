from dataclasses import dataclass


@dataclass(frozen=True)
class MatchThreshold:
    threshold_95: float
    threshold_99: float