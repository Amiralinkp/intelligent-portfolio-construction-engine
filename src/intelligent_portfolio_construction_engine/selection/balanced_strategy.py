from intelligent_portfolio_construction_engine.selection.base_strategy import BaseSelectionStrategy


class BalancedSelectionStrategy(BaseSelectionStrategy):


    def score(self, profile):

        f = profile.features

        score = (
            f.annual_return * 0.20
            + f.cagr * 0.20
            + f.sharpe_ratio * 0.20
            + f.sortino_ratio * 0.15
            + f.roc * 0.10
            + f.rsi * 0.05
            - f.volatility * 0.05
            - f.max_drawdown * 0.05)

        return score