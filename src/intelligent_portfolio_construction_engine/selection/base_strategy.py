from abc import ABC, abstractmethod
from intelligent_portfolio_construction_engine.models.asset_profile import AssetProfile
from intelligent_portfolio_construction_engine.config.policy import PolicyConfig


class BaseSelectionStrategy(ABC):

    def __init__(self, policy_config: PolicyConfig):

        self.policy_config = policy_config

    @abstractmethod
    def score(self, profile: AssetProfile):
        pass