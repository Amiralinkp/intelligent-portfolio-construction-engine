from intelligent_portfolio_construction_engine.selection.base_strategy import BaseSelectionStrategy


class RiskAverseSelectionStrategy(BaseSelectionStrategy):


    def score(self, profile):

        f = profile.features

        score = (
            f.sharpe_ratio * 0.30
            + f.sortino_ratio * 0.25
            + f.cagr * 0.15
            + f.annual_return * 0.10
            - f.volatility * 0.10
            - f.max_drawdown * 0.10)

        return score