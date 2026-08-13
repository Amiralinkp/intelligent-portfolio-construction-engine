from intelligent_portfolio_construction_engine.selection.return_focused_strategy import ReturnFocusedStrategy
from intelligent_portfolio_construction_engine.selection.balanced_strategy import BalancedSelectionStrategy
from intelligent_portfolio_construction_engine.selection.riskaverse_strategy import RiskAverseSelectionStrategy
from intelligent_portfolio_construction_engine.models.portfolio_config import PortfolioObjective

class StrategyFactory:

    def create(strategy_name):

        if strategy_name == PortfolioObjective.RETURN_FOCUSED:
            return ReturnFocusedStrategy()

        elif strategy_name == PortfolioObjective.BALANCED:
            return BalancedSelectionStrategy()

        elif strategy_name == PortfolioObjective.RISK_AVERSE:
            return RiskAverseSelectionStrategy()

        else:
            raise ValueError("The method is not in strategies")