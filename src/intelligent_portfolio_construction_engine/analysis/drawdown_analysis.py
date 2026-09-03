import pandas as pd
from intelligent_portfolio_construction_engine.models.drawdown_episode import DrawdownEpisode




def calculate_drawdown_series(asset_df):
    close = asset_df["Close"]

    peak = close.cummax()
    drawdown = (peak - close) / peak

    return peak, drawdown

def detect_drawdown_episodes(asset_df):
    peak, drawdown = calculate_drawdown_series(asset_df)

    episodes = []

    in_drawdown = False

    peak_date = None
    peak_price = None

    trough_date = None
    trough_price = None
    max_drawdown = 0.0

    for date, value in drawdown.items():

        if not in_drawdown and value > 0:
            in_drawdown = True

            peak_date = peak.index[drawdown.index.get_loc(date) - 1]

            peak_price = peak.loc[peak_date]

        if in_drawdown and value > max_drawdown:
            max_drawdown = value
            trough_date = date
            trough_price = asset_df.loc[date, "Close"]

        if in_drawdown and value == 0:

            recovery_date = date
            recovery_levels = calculate_recovery_levels(
                            asset_df=asset_df,
                            trough_date=trough_date,
                            trough_price=trough_price,
                            peak_price=peak_price)
            
    

            episodes.append(
                DrawdownEpisode(
                    peak_date=peak_date,
                    trough_date=trough_date,
                    recovery_date=recovery_date,
                    peak_price=peak_price,
                    trough_price=trough_price,
                    magnitude=max_drawdown,
                    drawdown_duration=(trough_date - peak_date).days,

                    recovery_25_date=recovery_levels[25],
                    recovery_25_time=(
                        (recovery_levels[25] - trough_date).days
                        if recovery_levels[25] is not None
                        else None),

                    recovery_50_date=recovery_levels[50],
                    recovery_50_time=(
                        (recovery_levels[50] - trough_date).days
                        if recovery_levels[50] is not None
                        else None),

                    recovery_70_date=recovery_levels[70],
                    recovery_70_time=(
                        (recovery_levels[70] - trough_date).days
                        if recovery_levels[70] is not None
                        else None),

                    recovery_90_date=recovery_levels[90],
                    recovery_90_time=(
                        (recovery_levels[90] - trough_date).days
                        if recovery_levels[90] is not None
                        else None),

                    recovery_100_date=recovery_levels[100],
                    recovery_100_time=(
                        (recovery_levels[100] - trough_date).days
                        if recovery_levels[100] is not None
                        else None)))

            in_drawdown = False

            peak_date = None
            peak_price = None

            trough_date = None
            trough_price = None

            max_drawdown = 0.0
            
    if in_drawdown:

        recovery_levels = calculate_recovery_levels(
            asset_df=asset_df, trough_date=trough_date, trough_price=trough_price, peak_price=peak_price)

        episodes.append(
            DrawdownEpisode(
                peak_date=peak_date,
                trough_date=trough_date,
                recovery_date=None,
                peak_price=peak_price,
                trough_price=trough_price,
                magnitude=max_drawdown,
                drawdown_duration=(trough_date - peak_date).days,

                recovery_25_date=recovery_levels[25],
                recovery_25_time=(
                    (recovery_levels[25] - trough_date).days
                    if recovery_levels[25] is not None
                    else None),

                recovery_50_date=recovery_levels[50],
                recovery_50_time=(
                    (recovery_levels[50] - trough_date).days
                    if recovery_levels[50] is not None
                    else None),

                recovery_70_date=recovery_levels[70],
                recovery_70_time=(
                    (recovery_levels[70] - trough_date).days
                    if recovery_levels[70] is not None
                    else None),

                recovery_90_date=recovery_levels[90],
                recovery_90_time=(
                    (recovery_levels[90] - trough_date).days
                    if recovery_levels[90] is not None
                    else None),

                recovery_100_date=recovery_levels[100],
                recovery_100_time=(
                    (recovery_levels[100] - trough_date).days
                    if recovery_levels[100] is not None
                    else None)))

    return episodes

def calculate_recovery_levels(asset_df, trough_date, trough_price, peak_price):

    recovery_levels = {
        25: None,
        50: None,
        70: None,
        90: None,
        100: None}

    drawdown_size = peak_price - trough_price

    for date, price in asset_df["Close"].loc[trough_date:].items():

        recovery_percentage = ((price - trough_price) / drawdown_size) * 100

        for level in recovery_levels:
            if (
                recovery_levels[level] is None
                and recovery_percentage >= level):

                recovery_levels[level] = date

    return recovery_levels