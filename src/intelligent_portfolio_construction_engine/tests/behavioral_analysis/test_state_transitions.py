
import pandas as pd
import pytest

from intelligent_portfolio_construction_engine.behavioral_analysis import (
    analyze_state_transitions,
)


@pytest.fixture
def behavioral_history():
    return pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(
                [
                    "2024-01-07",
                    "2024-01-14",
                    "2024-01-21",
                    "2024-01-28",
                    "2024-02-04",
                    "2024-01-07",
                    "2024-01-14",
                    "2024-01-21",
                ]
            ),
            "symbol": [
                "BTC", "BTC", "BTC", "BTC", "BTC",
                "ETH", "ETH", "ETH",
            ],
            "cluster_id": [0, 0, 1, 1, 0, 1, 1, 0],
            "model_version": [
                "v1", "v1", "v1", "v2", "v2",
                "v1", "v1", "v1",
            ],
        }
    )


def test_state_paths_are_sorted_and_segmented(
    behavioral_history,
):
    result = analyze_state_transitions(behavioral_history)

    btc_paths = result.asset_state_paths[
        result.asset_state_paths["symbol"] == "BTC"
    ]

    assert btc_paths["snapshot_date"].is_monotonic_increasing
    assert btc_paths["segment_id"].nunique() == 1


def test_gap_breaks_transition_continuity(
    behavioral_history,
):
    history = behavioral_history.copy()

    history.loc[
        (history["symbol"] == "BTC")
        & (history["snapshot_date"] == pd.Timestamp("2024-01-28")),
        "snapshot_date",
    ] = pd.Timestamp("2024-02-18")

    result = analyze_state_transitions(history)

    btc_events = result.transition_events[
        result.transition_events["symbol"] == "BTC"
    ]

    assert not (
        (btc_events["from_date"] == pd.Timestamp("2024-01-21"))
        & (btc_events["to_date"] == pd.Timestamp("2024-02-18"))
    ).any()


def test_transition_events_capture_cluster_changes(
    behavioral_history,
):
    result = analyze_state_transitions(behavioral_history)

    btc_events = result.transition_events[
        result.transition_events["symbol"] == "BTC"
    ]

    transition = btc_events[
        (btc_events["from_date"] == pd.Timestamp("2024-01-14"))
        & (btc_events["to_date"] == pd.Timestamp("2024-01-21"))
    ].iloc[0]

    assert transition["from_cluster"] == 0
    assert transition["to_cluster"] == 1
    assert transition["from_model_version"] == "v1"
    assert transition["to_model_version"] == "v1"


def test_model_version_change_does_not_split_same_state_run(
    behavioral_history,
):
    result = analyze_state_transitions(behavioral_history)

    btc_durations = result.state_durations[
        (result.state_durations["symbol"] == "BTC")
        & (result.state_durations["cluster_id"] == 1)
    ]

    assert len(btc_durations) == 1
    assert set(btc_durations.iloc[0]["model_versions"]) == {
        "v1",
        "v2",
    }


def test_transition_matrix_probabilities_are_normalized(
    behavioral_history,
):
    result = analyze_state_transitions(behavioral_history)

    probabilities = result.market_transition_probabilities

    for cluster_id in probabilities.index:
        row = probabilities.loc[cluster_id].dropna()

        if not row.empty:
            assert row.sum() == pytest.approx(1.0)


def test_asset_transition_matrices_are_available(
    behavioral_history,
):
    result = analyze_state_transitions(behavioral_history)

    assert set(result.asset_transition_matrices) == {"BTC", "ETH"}

    counts, probabilities = result.asset_transition_matrices["BTC"]

    assert counts.loc[0, 1] == 1
    assert probabilities.loc[0, 1] == pytest.approx(0.5)


def test_asset_summary_contains_persistence_ratios(
    behavioral_history,
):
    result = analyze_state_transitions(behavioral_history)

    btc_summary = result.asset_state_summary[
        result.asset_state_summary["symbol"] == "BTC"
    ]

    assert not btc_summary.empty
    assert btc_summary["persistence_ratio"].between(0, 1).all()
    assert btc_summary["persistence_ratio"].sum() == pytest.approx(1.0)


def test_market_summary_contains_expected_clusters(
    behavioral_history,
):
    result = analyze_state_transitions(behavioral_history)

    summary = result.market_state_summary

    assert set(summary["cluster_id"]) == {0, 1}
    assert (summary["asset_count"] >= 1).all()


def test_state_transitions_reject_duplicate_snapshots(
    behavioral_history,
):
    duplicated = pd.concat(
        [behavioral_history, behavioral_history.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError):
        analyze_state_transitions(duplicated)


def test_state_transitions_reject_missing_columns(
    behavioral_history,
):
    invalid = behavioral_history.drop(columns=["model_version"])

    with pytest.raises(ValueError):
        analyze_state_transitions(invalid)
