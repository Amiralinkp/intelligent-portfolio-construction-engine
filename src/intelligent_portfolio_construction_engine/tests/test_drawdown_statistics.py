from intelligent_portfolio_construction_engine.analysis.drawdown_statistics import calculate_median_recovery, calculate_weighted_recovery, calculate_drawdown_statistics
import pandas as pd 
from intelligent_portfolio_construction_engine.models.drawdown_episode import DrawdownEpisode
from intelligent_portfolio_construction_engine.analysis.drawdown_analysis import detect_drawdown_episodes



def test_median_rt():
    dates = pd.date_range(start="2025-01-01", periods=7, freq="D")

    data = pd.DataFrame(
        {
            "Close": [100, 70, 77.5, 85, 91, 97, 100],
        },
        index=dates,
    )

    episodes = detect_drawdown_episodes(data)

    result = calculate_median_recovery(episodes, 25)

    assert result == 1

def test_weighted_recovery():
    dates = pd.date_range(start="2025-01-01", periods=7, freq="D")

    data = pd.DataFrame(
        {
            "Close": [100, 70, 77.5, 85, 91, 97, 100],
        },
        index=dates,
    )

    episodes = detect_drawdown_episodes(data)

    result = calculate_weighted_recovery(episodes, 25)

    assert result == 1

def test_fin():

    dates = pd.date_range(start="2025-01-01", periods=7, freq="D")

    data = pd.DataFrame(
        {
            "Close": [100, 70, 77.5, 85, 91, 97, 100],
        },
        index=dates,
    )

    episodes = detect_drawdown_episodes(data)
    statistics = calculate_drawdown_statistics(episodes)


    assert statistics.drawdown_frequency == 1
    assert statistics.average_drawdown == 0.30
    assert statistics.max_drawdown == 0.30

    assert statistics.median_recovery_25 == 1
    assert statistics.median_recovery_50 == 2
    assert statistics.median_recovery_70 == 3
    assert statistics.median_recovery_90 == 4
    assert statistics.median_recovery_100 == 5

    assert statistics.recovery_success_rate == 100
    assert statistics.unrecovered_count == 0
    assert statistics.weighted_recovery_25 == 1
    assert statistics.weighted_recovery_50 == 2
    assert statistics.weighted_recovery_70 == 3
    assert statistics.weighted_recovery_90 == 4
    assert statistics.weighted_recovery_100 == 5
