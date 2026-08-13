from intelligent_portfolio_construction_engine.selection.base_strategy import BaseSelectionStrategy



class ReturnFocusedStrategy(BaseSelectionStrategy):

    def score(self, profile):

        f = profile.features

        positive = (
            f.annual_return * 0.25
            + f.cagr * 0.20
            + f.sharpe_ratio * 0.15
            + f.roc * 0.20
            + (100 - abs(50 - f.rsi)) / 100 * 0.10)

        negative = (
            f.volatility * 0.03
            + f.max_drawdown * 0.05
            + f.atr * 0.02)

        score = positive - negative
        return score