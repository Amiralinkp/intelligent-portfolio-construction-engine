from intelligent_portfolio_construction_engine.models.portfolio_config import PortfolioConfig




class PortfolioSelectionPolicy:

    def __init__(self, config: PortfolioConfig, profiles):

        self.config = config
        self.profiles = profiles


    def get_capital(self):
        

        if 0 < self.config.capital < 10000:
            return "small"
            

        elif 10000 <= self.config.capital <= 50000 :
            return "medium"

        elif self.config.capital > 50000: 
            return "large"

        else:
            raise ValueError("capital must be grater than 0 ")
        
    def get_asset_limits(self):

        capital_size = self.get_capital()

        if capital_size == "small":
            core_assets, optional_assets = 3, 1 

            return core_assets, optional_assets
        
        elif capital_size == "medium":
            core_assets, optional_assets = 4, 2

            return core_assets, optional_assets
        
        elif capital_size == "large":
            core_assets, optional_assets = 6, 2 

            return core_assets, optional_assets
        
        else:
            raise ValueError("error in capital_size")
        
    def select_core_assets(self):

        core_assets, _ = self.get_asset_limits()
        
        sectors = set()
        core_profiles = []

        for profile in self.profiles:

            sector = profile.sector

            if sector in sectors:
                continue

            core_profiles.append(profile)
            sectors.add(sector)

            if len(core_profiles) == core_assets:
                break

        return core_profiles


    def select_optional_assets(self, core_profiles):
        # core_scores = [profile.score for profile in core_profiles]
        # core_scores_mean = sum(core_scores) / len(core_scores)
        _, optional_assets = self.get_asset_limits()

        optional_profiles = []

        for profile in self.profiles:

            if profile in core_profiles:
                continue

            # if profile.score > core_scores_mean:

            optional_profiles.append(profile)

            if len(optional_profiles) >= optional_assets:
                break

        return optional_profiles
        






        

        


