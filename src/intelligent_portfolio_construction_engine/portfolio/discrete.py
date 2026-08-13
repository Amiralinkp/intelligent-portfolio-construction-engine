from intelligent_portfolio_construction_engine.models.asset_profile import AssetProfile
from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt.discrete_allocation import get_latest_prices
import pandas as pd


class DiscreteAllocator:

    def __init__(self, profiles, weights, capital):

        self.profiles = profiles
        self.weights = weights
        self.capital = capital

    def prepare_price_matrix(self):

        price_dict = {}

        for profile in self.profiles:

            historical_data = profile.historical_data

            if "Adj Close" in historical_data.columns:
                prices = historical_data["Adj Close"]
            else:
                prices = historical_data["Close"]

            price_dict[profile.symbol] = prices

        return pd.DataFrame(price_dict)

    def allocate(self):

        price_df = self.prepare_price_matrix()

        latest_prices = get_latest_prices(price_df)

        allocator = DiscreteAllocation(
            weights=self.weights,
            latest_prices=latest_prices,
            total_portfolio_value=self.capital)

        shares, cash = allocator.greedy_portfolio()

        return shares, cash