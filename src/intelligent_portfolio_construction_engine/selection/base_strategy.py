from abc import ABC, abstractmethod
from intelligent_portfolio_construction_engine.models.asset_profile import AssetProfile


class BaseSelectionStrategy(ABC):

    @abstractmethod

    def score(self, profile: AssetProfile) :
        pass