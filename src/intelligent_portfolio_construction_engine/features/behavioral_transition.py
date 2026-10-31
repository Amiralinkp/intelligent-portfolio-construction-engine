import pandas as pd
from pathlib import Path

def build_transition_history(
    behavioral_history: pd.DataFrame,
    max_gap_days: int = 10,
) -> pd.DataFrame:
        
    required_columns = [
        "snapshot_date",
        "symbol",
        "asset_class",
        "cluster_id",
    ]
    if max_gap_days < 1:
        raise ValueError("Maximum gap must be positive.")
    
    missing_columns = set(required_columns).difference(behavioral_history.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    history = behavioral_history[required_columns].copy()

    history["snapshot_date"] = pd.to_datetime(history["snapshot_date"])

    if history.duplicated(["symbol", "snapshot_date"]).any():
        raise ValueError("Duplicate asset/date snapshots found.")

    history = history.sort_values(["symbol", "snapshot_date"], ignore_index=True,)

    grouped = history.groupby("symbol", sort=False)

    history["previous_cluster"] = grouped["cluster_id"].shift(1)
    history["previous_date"] = grouped["snapshot_date"].shift(1)

    history = history.dropna(
        subset=["previous_cluster", "previous_date"]
    ).copy()

    history["previous_cluster"] = history[
        "previous_cluster"
    ].astype(int)

    history["cluster_id"] = history["cluster_id"].astype(int)

    history["transition"] = (
        history["previous_cluster"].astype(str)
        + "->"
        + history["cluster_id"].astype(str))

    history["days_between_snapshots"] = (
    history["snapshot_date"] - history["previous_date"]).dt.days

    history = history[
        history["days_between_snapshots"] <= max_gap_days].copy()

    history["transition"] = (
        history["previous_cluster"].astype(str)
        + "->"
        + history["cluster_id"].astype(str)
    )

    return history[
        [
            "symbol",
            "asset_class",
            "previous_date",
            "snapshot_date",
            "previous_cluster",
            "cluster_id",
            "transition",
            "days_between_snapshots",
        ]
    ]

def build_transition_matrix(
    transition_history: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = [
        "previous_cluster",
        "cluster_id",
    ]



    missing_columns = set(required_columns).difference(
        transition_history.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if transition_history.empty:
        raise ValueError("Transition history is empty.")

    if transition_history[required_columns].isna().any().any():
        raise ValueError("Transition data contains missing values.")

    transitions = transition_history.copy()

    matrix = pd.crosstab(
        index=transitions["previous_cluster"],
        columns=transitions["cluster_id"],
        normalize="index",
    )

    clusters = sorted(
        set(transitions["previous_cluster"])
        | set(transitions["cluster_id"])
    )

    matrix = matrix.reindex(
        index=clusters,
        columns=clusters,
        fill_value=0.0,
    )

    matrix.index.name = "from_cluster"
    matrix.columns.name = "to_cluster"

    return matrix


def build_state_duration(
    behavioral_history: pd.DataFrame,
    max_gap_days: int = 10,
) -> pd.DataFrame:
    required_columns = [
        "snapshot_date",
        "symbol",
        "asset_class",
        "cluster_id",
    ]

    missing_columns = set(required_columns).difference(
        behavioral_history.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if behavioral_history.empty:
        raise ValueError("Behavioral history is empty.")

    if max_gap_days < 1:
        raise ValueError("Maximum gap must be positive.")

    history = behavioral_history[required_columns].copy()

    history["snapshot_date"] = pd.to_datetime(
        history["snapshot_date"]
    )

    if history.duplicated(["symbol", "snapshot_date"]).any():
        raise ValueError("Duplicate asset/date snapshots found.")

    history = history.sort_values(
        ["symbol", "snapshot_date"],
        ignore_index=True,
    )

    grouped = history.groupby("symbol", sort=False)

    history["previous_cluster"] = grouped["cluster_id"].shift(1)
    history["previous_date"] = grouped["snapshot_date"].shift(1)

    history["days_between_snapshots"] = (
        history["snapshot_date"] - history["previous_date"]
    ).dt.days

    history["state_change"] = (
        history["cluster_id"] != history["previous_cluster"]
    ) | (
        history["days_between_snapshots"] > max_gap_days
    )

    history["state_change"] = history["state_change"].fillna(True)

    history["state_run"] = (
        history.groupby("symbol")["state_change"].cumsum()
    )

    duration = (
        history.groupby(
            ["symbol", "asset_class", "cluster_id", "state_run"],
            as_index=False,
        )
        .agg(
            start_date=("snapshot_date", "min"),
            end_date=("snapshot_date", "max"),
            snapshot_count=("snapshot_date", "count"),
        )
    )

    duration["duration_days"] = (
        duration["end_date"] - duration["start_date"]
    ).dt.days

    duration = duration.sort_values(
    ["symbol", "start_date"],
    ignore_index=True,)
    
    return duration.drop(columns="state_run")


def build_recurring_patterns(
    behavioral_history: pd.DataFrame,
    pattern_length: int = 3,
    min_occurrences: int = 2,
    max_gap_days: int = 10,
) -> pd.DataFrame:
    required_columns = [
        "snapshot_date",
        "symbol",
        "cluster_id",
    ]

    if max_gap_days < 1:
        raise ValueError("Maximum gap must be positive.")

    missing_columns = set(required_columns).difference(
        behavioral_history.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if behavioral_history.empty:
        raise ValueError("Behavioral history is empty.")

    if pattern_length < 2:
        raise ValueError("Pattern length must be at least 2.")

    if min_occurrences < 1:
        raise ValueError("Minimum occurrences must be positive.")

    history = behavioral_history[required_columns].copy()

    history["snapshot_date"] = pd.to_datetime(
        history["snapshot_date"]
    )

    if history.duplicated(["symbol", "snapshot_date"]).any():
        raise ValueError("Duplicate asset/date snapshots found.")

    history = history.sort_values(
        ["symbol", "snapshot_date"],
        ignore_index=True,
    )

    patterns = []

    for symbol, asset_history in history.groupby("symbol", sort=False):
        states = asset_history["cluster_id"].tolist()
        dates = asset_history["snapshot_date"].tolist()

        for index in range(len(states) - pattern_length + 1):
            window_dates = dates[index : index + pattern_length]

            gaps = [
                (window_dates[i] - window_dates[i - 1]).days
                for i in range(1, len(window_dates))
            ]

            if any(gap > max_gap_days for gap in gaps):
                continue

            pattern = tuple(
                states[index : index + pattern_length]
            )

            patterns.append(
                {
                    "symbol": symbol,
                    "pattern": pattern,
                    "start_date": window_dates[0],
                    "end_date": window_dates[-1],
                }
            )

    if not patterns:
        return pd.DataFrame(
            columns=[
                "pattern",
                "occurrences",
                "asset_count",
                "first_observed",
                "last_observed",
            ]
        )

    pattern_history = pd.DataFrame(patterns)

    summary = (
        pattern_history.groupby("pattern", as_index=False)
        .agg(
            occurrences=("pattern", "count"),
            asset_count=("symbol", "nunique"),
            first_observed=("start_date", "min"),
            last_observed=("end_date", "max"),
        )
    )

    summary = summary[
        summary["occurrences"] >= min_occurrences
    ]

    summary["pattern"] = summary["pattern"].apply(
        lambda states: " -> ".join(map(str, states))
    )

    return summary.sort_values(
        ["occurrences", "asset_count"],
        ascending=False,
        ignore_index=True,
    )


def analyze_behavioral_transitions(
    behavioral_history: pd.DataFrame,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)

    max_gap_days = 10

    transition_history = build_transition_history(
        behavioral_history,
        max_gap_days=max_gap_days,
    )

    transition_matrix = build_transition_matrix(
        transition_history,
    )

    state_duration = build_state_duration(
        behavioral_history,
        max_gap_days=max_gap_days,
    )

    recurring_patterns = build_recurring_patterns(
        behavioral_history=behavioral_history,
        pattern_length=3,
        min_occurrences=2,
        max_gap_days=max_gap_days,
    )

    outputs = {
        "transition_history": transition_history,
        "transition_matrix": transition_matrix,
        "state_duration": state_duration,
        "recurring_patterns": recurring_patterns,
    }

    for name, dataframe in outputs.items():
        dataframe.to_csv(
            output_dir / f"{name}.csv",
            index=name == "transition_matrix",
        )

    return outputs