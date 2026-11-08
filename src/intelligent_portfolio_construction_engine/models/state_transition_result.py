
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StateTransitionResult:
    asset_state_paths: pd.DataFrame
    transition_events: pd.DataFrame

    market_transition_counts: pd.DataFrame
    market_transition_probabilities: pd.DataFrame

    asset_transition_matrices: dict[
        str,
        tuple[pd.DataFrame, pd.DataFrame],
    ]

    state_durations: pd.DataFrame
    asset_state_summary: pd.DataFrame
    market_state_summary: pd.DataFrame
