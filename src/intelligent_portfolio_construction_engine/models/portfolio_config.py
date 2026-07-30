from dataclasses import dataclass
from enum import Enum


class PortfolioObjective(Enum):

    RISK_AVERSE = "risk_averse"
    BALANCED = "balanced"
    RETURN_FOCUSED = "return_focused"


@dataclass
class PortfolioConfig:

    capital: float
    
    objective: PortfolioObjective


