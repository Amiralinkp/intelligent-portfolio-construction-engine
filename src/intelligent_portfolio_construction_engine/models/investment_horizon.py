from enum import Enum
from dataclasses import dataclass

class InvestmentHorizon(str, Enum):
    
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class PortfolioObjective(str, Enum):

    RISK_AVERSE = "risk_averse"
    BALANCED = "balanced"
    RETURN_FOCUSED = "return_focused"


@dataclass(frozen=True)
class PortfolioPolicy:
    objective: PortfolioObjective
    horizon: InvestmentHorizon