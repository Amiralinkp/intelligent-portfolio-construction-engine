from dataclasses import dataclass

from intelligent_portfolio_construction_engine.models.drawdown_statistics import DrawdownStatistics
from intelligent_portfolio_construction_engine.models.historical_context import HistoricalContext



@dataclass
class HistoricalAnalysis:
    
    historical_context: HistoricalContext
    drawdown_statistics: DrawdownStatistics