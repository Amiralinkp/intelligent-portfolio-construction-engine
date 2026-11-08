import numpy as np
import pandas as pd
import pytest

from intelligent_portfolio_construction_engine.behavioral_analysis import (
    analyze_market_behavior_distribution,
)


@pytest.fixture
def market_history():
    return pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(
                [
                    "2024-01-07",
                    "2024-01-07",
                    "2024-01-07",
                    "2024-01-07",
                    "2024-01-14",
                    "2024-01-14",
                ]
            ),
            "symbol": [
                "BTC",
                "ETH",
                "GOLD",
                "SP500",
                "BTC",
                "ETH",
            ],
            "cluster_id": [0, 0, 0, 1, 1, 1],
        }
    )


def test_market_distribution_calculates_cluster_shares(
    market_history,
):
    result = analyze_market_behavior_distribution(
        market_history
    )

    first_date = result[
        result["snapshot_date"] == pd.Timestamp("2024-01-07")
    ]

    cluster_zero = first_date[
        first_date["cluster_id"] == 0
    ].iloc[0]

    cluster_one = first_date[
        first_date["cluster_id"] == 1
    ].iloc[0]

    assert cluster_zero["asset_count"] == 3
    assert cluster_one["asset_count"] == 1

    assert cluster_zero["cluster_share"] == pytest.approx(0.75)
    assert cluster_one["cluster_share"] == pytest.approx(0.25)


def test_market_distribution_calculates_normalized_entropy(
    market_history,
):
    result = analyze_market_behavior_distribution(
        market_history
    )

    first_date = result[
        result["snapshot_date"] == pd.Timestamp("2024-01-07")
    ]

    entropy = first_date["normalized_entropy"].iloc[0]

    expected_entropy = (
        -(0.75 * np.log(0.75) + 0.25 * np.log(0.25))
        / np.log(2)
    )

    assert entropy == pytest.approx(expected_entropy)


def test_market_distribution_detects_single_cluster(
    market_history,
):
    single_cluster = market_history.copy()
    single_cluster["cluster_id"] = 0

    result = analyze_market_behavior_distribution(
        single_cluster
    )

    assert (result["normalized_entropy"] == 0.0).all()
    assert (result["dominant_cluster_share"] == 1.0).all()


def test_market_distribution_rejects_empty_input(
    market_history,
):
    with pytest.raises(ValueError):
        analyze_market_behavior_distribution(
            market_history.iloc[0:0]
        )


def test_market_distribution_rejects_duplicate_snapshots(
    market_history,
):
    duplicated = pd.concat(
        [market_history, market_history.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError):
        analyze_market_behavior_distribution(duplicated)


def test_market_distribution_rejects_missing_values(
    market_history,
):
    invalid = market_history.copy()
    invalid.loc[0, "cluster_id"] = np.nan

    with pytest.raises(ValueError):
        analyze_market_behavior_distribution(invalid)