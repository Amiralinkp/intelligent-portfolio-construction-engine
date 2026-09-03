import pandas as pd

from intelligent_portfolio_construction_engine.analysis.drawdown_analysis import detect_drawdown_episodes



def test_drawdown_episode():

    dates = pd.date_range(
        start="2025-01-01",
        periods=7,
        freq="D")

    asset_df = pd.DataFrame(
        {"Close": [100, 70, 77.5, 85, 91, 97, 100]},
        index=dates )

    episodes = detect_drawdown_episodes(asset_df)

    assert len(episodes) == 1
    assert episodes[0].peak_price == 100
    assert episodes[0].trough_price == 70
    assert episodes[0].recovery_25_date is not None
    assert episodes[0].recovery_50_date is not None
    assert episodes[0].recovery_70_date is not None
    assert episodes[0].recovery_90_date is not None
    assert episodes[0].recovery_100_date is not None
    assert episodes[0].recovery_25_date == pd.Timestamp("2025-01-03")
    assert episodes[0].recovery_50_date == pd.Timestamp("2025-01-04")
    assert episodes[0].recovery_70_date == pd.Timestamp("2025-01-05")
    assert episodes[0].recovery_90_date == pd.Timestamp("2025-01-06")
    assert episodes[0].recovery_100_date == pd.Timestamp("2025-01-07")
    assert episodes[0].recovery_25_time == 1
    assert episodes[0].recovery_50_time == 2
    assert episodes[0].recovery_70_time == 3
    assert episodes[0].recovery_90_time == 4
    assert episodes[0].recovery_100_time == 5

def test_unrecovered_drawdown_episode():

    dates = pd.date_range(
            start="2025-01-01",
            periods=5,
            freq="D")

    asset_df = pd.DataFrame(
        {"Close":[100, 70, 77.5, 85, 91]},
        index=dates)
    episodes = detect_drawdown_episodes(asset_df)

    assert len(episodes) == 1
    assert episodes[0].peak_price == 100
    assert episodes[0].trough_price == 70
    assert episodes[0].recovery_25_date is not None
    assert episodes[0].recovery_50_date is not None
    assert episodes[0].recovery_70_date is not None
    assert episodes[0].recovery_90_date is None
    assert episodes[0].recovery_100_date is None
    assert episodes[0].magnitude == 0.30