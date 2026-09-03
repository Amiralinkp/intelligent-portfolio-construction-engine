from statistics import median
from intelligent_portfolio_construction_engine.models.drawdown_statistics import DrawdownStatistics


def calculate_drawdown_frequency(episodes):
    return len(episodes)

def calculate_average_drawdown(episodes):
    total = 0

    for episode in episodes:
        magnitude = episode.magnitude
        total = total+magnitude

    return total/len(episodes)

def calculate_max_drawdown(episodes):
    return max(episode.magnitude for episode in episodes)

def calculate_median_recovery(episodes, level):
    recovery_times =[getattr(episode, f"recovery_{level}_time") for episode in episodes
                     if getattr(episode, f"recovery_{level}_time") is not None]
    
    return median(recovery_times) if recovery_times else None

def calculate_weighted_recovery(episodes, level):

    res_up = []
    res_down = []

    for episode in episodes:
        magnitude = episode.magnitude
        recovery = getattr(episode, f"recovery_{level}_time")

        if recovery is not None:
            res = magnitude*recovery
            res_up.append(res)
            res_down.append(magnitude)

    if not res_down:
        raise ValueError("No valid recovery data available for this recovery level.")

    return sum(res_up)/sum (res_down)


def calculate_unrecovered_count(episodes):

    unrecovered = []

    for episode in episodes:
        if episode.recovery_100_time is None:

            unrecovered.append(episode)

    return len(unrecovered)

def calculate_recovery_success_rate(episodes):

    recovered = 0

    for episode in episodes:
        if episode.recovery_100_time is not None:

            recovered += 1

    return (recovered / len(episodes)) * 100


def calculate_drawdown_statistics(episodes):
    drawdown_frequency = calculate_drawdown_frequency(episodes)
    average_drawdown = calculate_average_drawdown(episodes)
    max_drawdown = calculate_max_drawdown(episodes)
    median_recovery_25 = calculate_median_recovery(episodes, 25)
    median_recovery_50 = calculate_median_recovery(episodes, 50)
    median_recovery_70 = calculate_median_recovery(episodes, 70)
    median_recovery_90 = calculate_median_recovery(episodes, 90)
    median_recovery_100 = calculate_median_recovery(episodes, 100)
    weighted_recovery_25 = calculate_weighted_recovery(episodes, 25)
    weighted_recovery_50 = calculate_weighted_recovery(episodes, 50)
    weighted_recovery_70 = calculate_weighted_recovery(episodes, 70)
    weighted_recovery_90 = calculate_weighted_recovery(episodes, 90)
    weighted_recovery_100 = calculate_weighted_recovery(episodes, 100)
    recovery_success_rate = calculate_recovery_success_rate(episodes)
    unrecovered_count = calculate_unrecovered_count(episodes)

    return DrawdownStatistics(
        drawdown_frequency=drawdown_frequency,
        average_drawdown=average_drawdown,
        max_drawdown=max_drawdown,
        median_recovery_25=median_recovery_25,
        median_recovery_50=median_recovery_50,
        median_recovery_70=median_recovery_70,
        median_recovery_90=median_recovery_90,
        median_recovery_100=median_recovery_100,
        weighted_recovery_25=weighted_recovery_25,
        weighted_recovery_50=weighted_recovery_50,
        weighted_recovery_70=weighted_recovery_70,
        weighted_recovery_90=weighted_recovery_90,
        weighted_recovery_100=weighted_recovery_100,
        recovery_success_rate = recovery_success_rate,
        unrecovered_count = unrecovered_count)