from intelligent_portfolio_construction_engine.selection.return_focused_strategy import ReturnFocusedStrategy
from intelligent_portfolio_construction_engine.selection.balanced_strategy import BalancedSelectionStrategy
from intelligent_portfolio_construction_engine.selection.riskaverse_strategy import RiskAverseSelectionStrategy
from intelligent_portfolio_construction_engine.models.portfolio_config import PortfolioObjective
from intelligent_portfolio_construction_engine.config.policy import PolicyConfig

class StrategyFactory:

    @staticmethod
    def create(strategy_name: PortfolioObjective, policy_config: PolicyConfig):

        if strategy_name == PortfolioObjective.RETURN_FOCUSED:
            return ReturnFocusedStrategy(policy_config)

        elif strategy_name == PortfolioObjective.BALANCED:
            return BalancedSelectionStrategy(policy_config)

        elif strategy_name == PortfolioObjective.RISK_AVERSE:
            return RiskAverseSelectionStrategy(policy_config)

        else:
            raise ValueError("The method is not in strategies")