from collections.abc import Sequence

import pandas as pd

from ..models.behavioral_analysis_result import (
    BehavioralAnalysisResult,
)
from .behavioral_analyzer import (
    analyze_cluster_characteristics,
)
from .market_behavior_analyzer import (
    analyze_market_behavior_distribution,
)
from .state_transition_analyzer import (
    analyze_state_transitions,
)


def analyze_behavioral_history(
    behavioral_history: pd.DataFrame,
    expected_interval: str = "7D",
    cluster_ids: Sequence[int] | None = None,
) -> BehavioralAnalysisResult:
    cluster_characteristics = analyze_cluster_characteristics(
        behavioral_history
    )

    state_transition = analyze_state_transitions(
        behavioral_history=behavioral_history,
        expected_interval=expected_interval,
        cluster_ids=cluster_ids,
    )

    market_behavior_distribution = (
        analyze_market_behavior_distribution(
            behavioral_history
        )
    )

    return BehavioralAnalysisResult(
        cluster_characteristics=cluster_characteristics,
        state_transition=state_transition,
        market_behavior_distribution=market_behavior_distribution,
    )