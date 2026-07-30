import pandas as pd
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import EfficientFrontier
from intelligent_portfolio_construction_engine.models.portfolio_config import PortfolioObjective

class WeightAllocator:

    def __init__(self, profiles, config):

        self.profiles = profiles
        self.config = config

    def allocate(self):...



    def prepare_price_matrix(self):

        price_dict = {}
        for profile in self.profiles:

            historical_data = profile.historical_data

            if "Adj Close" in historical_data.columns:
                prices = historical_data["Adj Close"]
            else:
                prices = historical_data["Close"]

            price_dict[profile.symbol] = prices
        price_df = pd.DataFrame(price_dict)

        return price_df
    
    def prepare_return_matrix(self):

        price_df = self.prepare_price_matrix()
        return_matrix = price_df.pct_change().dropna()

        return return_matrix
        


    def expected_returns(self):

        price_df = self.prepare_price_matrix()
        if self.config.expected_return_method == "capm":

            mu = expected_returns.capm_return(price_df)
        
        elif self.config.expected_return_method == "mean":

            mu = expected_returns.mean_historical_return(price_df)

        else: 
            raise ValueError("Unknown method")

        return mu
    
    def covariance_matrix(self):

        price_df = self.prepare_price_matrix()
        covariance = risk_models.CovarianceShrinkage(price_df).ledoit_wolf()

        return covariance


    def optimize(self):

        mu = self.expected_returns()

        covariance = self.covariance_matrix()

        ef = EfficientFrontier(mu, covariance)

        if self.config.objective == PortfolioObjective.RISK_AVERSE:
            ef.min_volatility()

        elif self.config.objective == PortfolioObjective.BALANCED:
            ef.max_sharpe()

        elif self.config.objective == PortfolioObjective.RETURN_FOCUSED:
            ef.max_quadratic_utility()

        return ef



