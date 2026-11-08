from dataclasses import dataclass

import pandas as pd

from .state_transition_result import StateTransitionResult


@dataclass(frozen=True)
class BehavioralAnalysisResult:
    cluster_characteristics: pd.DataFrame
    state_transition: StateTransitionResult
    market_behavior_distribution: pd.DataFrame