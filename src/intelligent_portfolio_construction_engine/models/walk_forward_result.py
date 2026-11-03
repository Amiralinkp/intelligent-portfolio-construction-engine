from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from intelligent_portfolio_construction_engine.models.cluster_alignment_result import (
    ClusterAlignmentResult,
)


@dataclass
class WalkForwardResult:
    behavioral_history: pd.DataFrame
    model_versions: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    alignment_results: dict[str, ClusterAlignmentResult] = field(
        default_factory=dict)