from intelligent_portfolio_construction_engine.models.portfolio_config import PortfolioConfig
from intelligent_portfolio_construction_engine.portfolio.selection_policy import PortfolioSelectionPolicy
from intelligent_portfolio_construction_engine.portfolio.weight_allocator import WeightAllocator
from intelligent_portfolio_construction_engine.portfolio.discrete import DiscreteAllocator

class PortfolioBuilder:


    def __init__(self, profiles, config):

        self.profiles = profiles
        self.config = config

    def build(self):

        policy = PortfolioSelectionPolicy(config=self.config, profiles=self.profiles)
        
        core_assets = policy.select_core_assets()
        optional_assets = policy.select_optional_assets(core_assets)
        final_assets = core_assets + optional_assets

        allocator = WeightAllocator(
        profiles=final_assets,
        config=self.config)

        weights, metrics = allocator.allocate()

        discrete_allocator = DiscreteAllocator(
        profiles=final_assets,
        weights=weights,
        capital=self.config.capital)

        shares, cash = discrete_allocator.allocate()
        
        return final_assets, weights, shares, cash, metrics