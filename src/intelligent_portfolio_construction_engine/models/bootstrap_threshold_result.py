from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapThresholdResult:
    threshold_95: float
    threshold_99: float