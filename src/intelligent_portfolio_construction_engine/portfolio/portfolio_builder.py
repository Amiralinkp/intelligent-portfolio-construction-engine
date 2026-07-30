from intelligent_portfolio_construction_engine.models.portfolio_config import PortfolioConfig
from intelligent_portfolio_construction_engine.portfolio.selection_policy import PortfolioSelectionPolicy

class PortfolioBuilder:


    def __init__(self, profiles, config):

        self.profiles = profiles
        self.config = config

    def build(self):

        policy = PortfolioSelectionPolicy(config=self.config, profiles=self.profiles)
        
        core_assets = policy.select_core_assets()
        optional_assets = policy.select_optional_assets(core_assets)
        final_assets = core_assets + optional_assets

        return final_assets
    