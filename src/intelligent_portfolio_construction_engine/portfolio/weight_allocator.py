import pandas as pd
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import EfficientFrontier
from intelligent_portfolio_construction_engine.models.portfolio_config import PortfolioObjective
from intelligent_portfolio_construction_engine.models.portfolio_metrics import PortfolioMetrics
from intelligent_portfolio_construction_engine.portfolio.selection_policy import PortfolioSelectionPolicy

class WeightAllocator:

    def __init__(self, profiles, config):

        self.profiles = profiles
        self.config = config

    def allocate(self):

        ef = self.optimize()
        weights = ef.clean_weights()
        expected_return, volatility, sharpe_ratio = ef.portfolio_performance()

        metrics = PortfolioMetrics(
        expected_return=expected_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio)

        return weights, metrics



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
        min_weight, max_weight = self.get_weight_limits()


        ef.add_constraint(lambda w: w >= min_weight)
        ef.add_constraint(lambda w: w <= max_weight)


        if self.config.objective == PortfolioObjective.RISK_AVERSE:
            ef.min_volatility()

        elif self.config.objective == PortfolioObjective.BALANCED:
            ef.max_sharpe()

        elif self.config.objective == PortfolioObjective.RETURN_FOCUSED:
            ef.max_quadratic_utility()


        return ef
    
    def get_weight_limits(self): 

        policy = PortfolioSelectionPolicy(profiles=self.profiles, config=self.config)

        budget = policy.get_capital()

        if budget=="small" : 
            min_weight = 0.15
            max_weight = 0.45

        elif budget == "medium":
            min_weight = 0.1
            max_weight = 0.35

        elif budget=="large":
            min_weight = 0.05
            max_weight = 0.25

        else:
            raise ValueError("weight_allactor is not responding")

        return min_weight, max_weight



