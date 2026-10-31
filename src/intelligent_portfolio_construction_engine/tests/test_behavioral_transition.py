
import pandas as pd
import pytest

from intelligent_portfolio_construction_engine.features.behavioral_transition import (
    analyze_behavioral_transitions,
    build_recurring_patterns,
    build_state_duration,
    build_transition_history,
    build_transition_matrix,
)


@pytest.fixture
def behavioral_history():
    dates = pd.date_range("2024-01-01", periods=5, freq="7D")

    return pd.DataFrame(
        {
            "snapshot_date": list(dates) + list(dates),
            "symbol": ["BTC-USD"] * 5 + ["ETH-USD"] * 5,
            "asset_class": ["crypto"] * 10,
            "cluster_id": [1, 1, 2, 1, 2, 2, 3, 3, 1, 2],
        }
    )


def test_build_transition_history(behavioral_history):
    result = build_transition_history(behavioral_history)

    assert len(result) == 8
    assert "previous_cluster" in result.columns
    assert "transition" in result.columns
    assert "days_between_snapshots" in result.columns

    btc_transitions = result[
        result["symbol"] == "BTC-USD"
    ]

    assert btc_transitions["transition"].tolist() == [
        "1->1",
        "1->2",
        "2->1",
        "1->2",
    ]


def test_transition_history_excludes_large_gaps():
    history = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(
                ["2024-01-01", "2024-01-08", "2024-01-22"]
            ),
            "symbol": ["BTC-USD"] * 3,
            "asset_class": ["crypto"] * 3,
            "cluster_id": [1, 2, 3],
        }
    )

    result = build_transition_history(
        history,
        max_gap_days=10,
    )

    assert len(result) == 1
    assert result.iloc[0]["transition"] == "1->2"


def test_build_transition_matrix(behavioral_history):
    transitions = build_transition_history(behavioral_history)

    matrix = build_transition_matrix(transitions)

    assert matrix.loc[2, 1] == pytest.approx(0.5)
    assert matrix.loc[2, 2] == pytest.approx(0.0)
    assert matrix.loc[2, 3] == pytest.approx(0.5)

    assert matrix.loc[3, 1] == pytest.approx(0.5)
    assert matrix.loc[3, 2] == pytest.approx(0.0)
    assert matrix.loc[3, 3] == pytest.approx(0.5)

def test_build_state_duration(behavioral_history):
    result = build_state_duration(behavioral_history)

    btc_duration = result[
        result["symbol"] == "BTC-USD"
    ].reset_index(drop=True)

    assert btc_duration["cluster_id"].tolist() == [1, 2, 1, 2]
    assert btc_duration["snapshot_count"].tolist() == [2, 1, 1, 1]
    assert btc_duration["duration_days"].tolist() == [7, 0, 0, 0]


def test_state_duration_splits_after_large_gap():
    history = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(
                ["2024-01-01", "2024-01-08", "2024-01-22"]
            ),
            "symbol": ["BTC-USD"] * 3,
            "asset_class": ["crypto"] * 3,
            "cluster_id": [1, 1, 1],
        }
    )

    result = build_state_duration(
        history,
        max_gap_days=10,
    )

    assert result["snapshot_count"].tolist() == [2, 1]


def test_build_recurring_patterns():
    dates = pd.date_range("2024-01-01", periods=5, freq="7D")

    history = pd.DataFrame(
        {
            "snapshot_date": dates,
            "symbol": ["BTC-USD"] * 5,
            "cluster_id": [1, 2, 1, 2, 1],
        }
    )

    result = build_recurring_patterns(
        behavioral_history=history,
        pattern_length=2,
        min_occurrences=2,
    )

    patterns = dict(
        zip(result["pattern"], result["occurrences"])
    )

    assert patterns["1 -> 2"] == 2
    assert patterns["2 -> 1"] == 2


def test_recurring_patterns_exclude_large_gaps():
    history = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(
                ["2024-01-01", "2024-01-08", "2024-01-22"]
            ),
            "symbol": ["BTC-USD"] * 3,
            "cluster_id": [1, 2, 3],
        }
    )

    result = build_recurring_patterns(
        behavioral_history=history,
        pattern_length=3,
        min_occurrences=1,
        max_gap_days=10,
    )

    assert result.empty


def test_analyze_behavioral_transitions(
    behavioral_history,
    tmp_path,
):
    outputs = analyze_behavioral_transitions(
        behavioral_history=behavioral_history,
        output_dir=tmp_path,
    )

    expected_files = [
        "transition_history.csv",
        "transition_matrix.csv",
        "state_duration.csv",
        "recurring_patterns.csv",
    ]

    for filename in expected_files:
        assert (tmp_path / filename).exists()

    assert set(outputs) == {
        "transition_history",
        "transition_matrix",
        "state_duration",
        "recurring_patterns",
    }


def test_duplicate_asset_date_raises_error(behavioral_history):
    duplicated = pd.concat(
        [behavioral_history, behavioral_history.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="Duplicate"):
        build_transition_history(duplicated)


def test_empty_transition_matrix_raises_error():
    empty_history = pd.DataFrame(
        columns=["previous_cluster", "cluster_id"]
    )

    with pytest.raises(ValueError, match="empty"):
        build_transition_matrix(empty_history)


def test_invalid_pattern_length_raises_error(behavioral_history):
    with pytest.raises(ValueError, match="at least 2"):
        build_recurring_patterns(
            behavioral_history,
            pattern_length=1,
        )
