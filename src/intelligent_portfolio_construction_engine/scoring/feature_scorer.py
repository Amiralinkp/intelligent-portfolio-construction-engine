from intelligent_portfolio_construction_engine.models.asset_profile import AssetProfile
from intelligent_portfolio_construction_engine.selection.strategy_factory import StrategyFactory



class FeatureScorer:

    def __init__(self, profiles, config):
        
        self.profiles = profiles
        self.config = config

    def score(self):

        strategy = StrategyFactory.create(self.config.objective)
        for profile in self.profiles:

            profile.score = strategy.score(profile)

        ranked_profiles = sorted(self.profiles, key=lambda profile : profile.score, reverse=True)

        return ranked_profiles