
import pandas as pd
from intelligent_portfolio_construction_engine.models.state_transition_result import (
    StateTransitionResult,
)

REQUIRED_COLUMNS = [
    "snapshot_date",
    "symbol",
    "cluster_id",
    "model_version",
]


def build_asset_state_paths(behavioral_history, expected_interval: str | pd.Timedelta = "7D"):
    missing_columns = set(REQUIRED_COLUMNS) - set(behavioral_history.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if behavioral_history.empty:
        raise ValueError("Behavioral history cannot be empty.")

    interval = pd.Timedelta(expected_interval)

    if interval <= pd.Timedelta(0):
        raise ValueError("Expected interval must be positive.")

    data = behavioral_history.copy()

    data["snapshot_date"] = pd.to_datetime(
        data["snapshot_date"]
    )

    if data[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(
            "Required columns cannot contain missing values."
        )

    if data.duplicated(["snapshot_date", "symbol"]).any():
        raise ValueError("Duplicate asset snapshots detected.")

    data = data.sort_values(
        ["symbol", "snapshot_date"]
    ).reset_index(drop=True)

    grouped_dates = data.groupby(
        "symbol",
        sort=False,
    )["snapshot_date"]

    data["segment_id"] = (
        grouped_dates.transform(
            lambda dates: dates.diff().gt(interval).cumsum()
        ).astype("int64")
    )

    data["snapshot_index"] = (
        data.groupby(
            ["symbol", "segment_id"],
            sort=False,
        ).cumcount()
    )

    return data


TRANSITION_COLUMNS = [
    "symbol",
    "from_date",
    "to_date",
    "from_cluster",
    "to_cluster",
    "from_model_version",
    "to_model_version",
    "interval_days",
]


def extract_transition_events(asset_state_paths):
    required_columns = {
        "snapshot_date",
        "symbol",
        "cluster_id",
        "model_version",
        "segment_id",
    }

    missing_columns = required_columns - set(
        asset_state_paths.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if asset_state_paths.empty:
        raise ValueError("Asset state paths cannot be empty.")

    data = asset_state_paths.copy()

    data["snapshot_date"] = pd.to_datetime(
        data["snapshot_date"]
    )

    if data[list(required_columns)].isna().any().any():
        raise ValueError(
            "Required transition columns cannot contain missing values."
        )

    data = data.sort_values(
        ["symbol", "segment_id", "snapshot_date"]
    ).reset_index(drop=True)

    grouped = data.groupby(
        ["symbol", "segment_id"],
        sort=False,
    )

    events = pd.DataFrame(
        {
            "symbol": data["symbol"],
            "from_date": data["snapshot_date"],
            "to_date": grouped["snapshot_date"].shift(-1),
            "from_cluster": data["cluster_id"],
            "to_cluster": grouped["cluster_id"].shift(-1),
            "from_model_version": data["model_version"],
            "to_model_version": grouped["model_version"].shift(-1),
        }
    )

    events = events.dropna(
        subset=["to_date"]
    ).copy()

    events["interval_days"] = (
        events["to_date"] - events["from_date"]
    ).dt.total_seconds() / 86400

    return events[TRANSITION_COLUMNS].reset_index(drop=True)




def build_transition_matrices(transition_events, cluster_ids: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_columns = {
        "from_cluster",
        "to_cluster",
    }

    missing_columns = required_columns - set(
        transition_events.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if transition_events.empty and cluster_ids is None:
        raise ValueError(
            "Cluster IDs are required when no transitions exist."
        )

    if cluster_ids is None:
        cluster_ids = sorted(
            set(transition_events["from_cluster"])
            | set(transition_events["to_cluster"])
        )

    if not cluster_ids:
        raise ValueError("Cluster IDs cannot be empty.")

    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("Cluster IDs must be unique.")

    unknown_clusters = (
        set(transition_events["from_cluster"])
        | set(transition_events["to_cluster"])
    ) - set(cluster_ids)

    if unknown_clusters:
        raise ValueError(
            f"Unknown cluster IDs: {sorted(unknown_clusters)}"
        )

    counts = (
        transition_events.groupby(
            ["from_cluster", "to_cluster"]
        )
        .size()
        .unstack(fill_value=0)
        .reindex(
            index=cluster_ids,
            columns=cluster_ids,
            fill_value=0,
        )
    )

    probabilities = counts.div(
        counts.sum(axis=1).replace(0, float("nan")),
        axis=0,
    )

    return counts, probabilities



def build_state_duration_summaries(asset_state_paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_columns = {
        "snapshot_date",
        "symbol",
        "cluster_id",
        "model_version",
        "segment_id",
    }

    missing_columns = required_columns - set(
        asset_state_paths.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if asset_state_paths.empty:
        raise ValueError("Asset state paths cannot be empty.")

    data = asset_state_paths.copy()

    data["snapshot_date"] = pd.to_datetime(
        data["snapshot_date"]
    )

    if data[list(required_columns)].isna().any().any():
        raise ValueError(
            "Required columns cannot contain missing values."
        )

    if data.duplicated(["snapshot_date", "symbol"]).any():
        raise ValueError("Duplicate asset snapshots detected.")

    data = data.sort_values(
        ["symbol", "segment_id", "snapshot_date"]
    ).reset_index(drop=True)

    asset_segments = data.groupby(
        ["symbol", "segment_id"],
        sort=False,
    )

    data["episode_id"] = (
        asset_segments["cluster_id"]
        .transform(lambda clusters: clusters.ne(clusters.shift()).cumsum())
        .astype("int64")
    )

    episodes = (
        data.groupby(
            ["symbol", "segment_id", "episode_id", "cluster_id"],
            sort=True,
        )
        .agg(
            start_date=("snapshot_date", "min"),
            end_date=("snapshot_date", "max"),
            duration_snapshots=("snapshot_date", "size"),
            start_model_version=("model_version", "first"),
            end_model_version=("model_version", "last"),
        )
        .reset_index()
    )

    episodes["elapsed_days"] = (
        episodes["end_date"] - episodes["start_date"]
    ).dt.total_seconds() / 86400

    def summarize_duration(
        frame: pd.DataFrame,
        group_columns: list[str],
    ) -> pd.DataFrame:
        return (
            frame.groupby(group_columns, sort=True)
            .agg(
                episode_count=("episode_id", "size"),
                total_snapshots=("duration_snapshots", "sum"),
                mean_duration_snapshots=(
                    "duration_snapshots", "mean"
                ),
                median_duration_snapshots=(
                    "duration_snapshots", "median"
                ),
                max_duration_snapshots=(
                    "duration_snapshots", "max"
                ),
                mean_elapsed_days=("elapsed_days", "mean"),
            )
            .reset_index()
        )

    asset_summary = summarize_duration(
        episodes,
        ["symbol", "cluster_id"],
    )

    market_summary = summarize_duration(
        episodes,
        ["cluster_id"],
    )

    return episodes, asset_summary, market_summary



def build_asset_transition_matrices(transition_events: pd.DataFrame, cluster_ids: list[int] | None = None) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    required_columns = {
        "symbol",
        "from_cluster",
        "to_cluster",
    }

    missing_columns = required_columns - set(
        transition_events.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if transition_events.empty:
        return {}

    if transition_events["symbol"].isna().any():
        raise ValueError("Asset symbols cannot be missing.")

    results = {}

    for symbol, asset_events in transition_events.groupby(
        "symbol",
        sort=True,
    ):
        counts, probabilities = build_transition_matrices(
            asset_events,
            cluster_ids=cluster_ids,
        )

        results[symbol] = (counts, probabilities)

    return results



def calculate_state_duration(asset_state_paths) -> pd.DataFrame:
    required_columns = {
        "symbol",
        "snapshot_date",
        "cluster_id",
        "model_version",
        "segment_id",
    }

    missing_columns = required_columns - set(
        asset_state_paths.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    output_columns = [
        "symbol",
        "segment_id",
        "cluster_id",
        "entry_date",
        "exit_date",
        "duration_snapshots",
        "duration_days",
        "model_versions",
    ]

    if asset_state_paths.empty:
        return pd.DataFrame(columns=output_columns)

    data = asset_state_paths.copy()

    data["snapshot_date"] = pd.to_datetime(
        data["snapshot_date"]
    )

    if data[list(required_columns)].isna().any().any():
        raise ValueError(
            "Required duration columns cannot contain missing values."
        )

    if data.duplicated(["symbol", "snapshot_date"]).any():
        raise ValueError("Duplicate asset snapshots detected.")

    data = data.sort_values(
        ["symbol", "segment_id", "snapshot_date"]
    ).reset_index(drop=True)

    group_columns = ["symbol", "segment_id"]

    state_change = (
        data.groupby(group_columns, sort=False)["cluster_id"]
        .transform(lambda values: values.ne(values.shift()))
    )

    data["state_run_id"] = (
        state_change.groupby(
            [data["symbol"], data["segment_id"]],
            sort=False,
        ).cumsum()
    )

    duration = (
        data.groupby(
            [
                "symbol",
                "segment_id",
                "state_run_id",
                "cluster_id",
            ],
            sort=True,
            as_index=False,
        )
        .agg(
            entry_date=("snapshot_date", "min"),
            exit_date=("snapshot_date", "max"),
            duration_snapshots=("snapshot_date", "size"),
            model_versions=(
                "model_version",
                lambda versions: tuple(
                    dict.fromkeys(versions)
                ),
            ),
        )
    )

    duration["duration_days"] = (
        duration["duration_snapshots"] * 7
    )

    return duration[output_columns]




def build_asset_state_summary(state_durations) -> pd.DataFrame:
    required_columns = {
        "symbol",
        "cluster_id",
        "duration_snapshots",
    }

    missing_columns = required_columns - set(
        state_durations.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if state_durations.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "cluster_id",
                "state_frequency",
                "total_snapshots",
                "mean_duration",
                "median_duration",
                "max_duration",
                "persistence_ratio",
            ]
        )

    data = state_durations.copy()

    if data[list(required_columns)].isna().any().any():
        raise ValueError(
            "Required summary columns cannot contain missing values."
        )

    asset_totals = data.groupby("symbol")[
        "duration_snapshots"
    ].transform("sum")

    summary = (
        data.groupby(
            ["symbol", "cluster_id"],
            sort=True,
        )
        .agg(
            state_frequency=("duration_snapshots", "size"),
            total_snapshots=("duration_snapshots", "sum"),
            mean_duration=("duration_snapshots", "mean"),
            median_duration=("duration_snapshots", "median"),
            max_duration=("duration_snapshots", "max"),
        )
        .reset_index()
    )

    summary["persistence_ratio"] = (
        summary["total_snapshots"]
        / summary["symbol"].map(
            data.assign(
                total=data["duration_snapshots"]
            )
            .groupby("symbol")["total"]
            .sum()
        )
    )

    return summary



def build_market_state_summary(state_durations) -> pd.DataFrame:
    required_columns = {
        "symbol",
        "cluster_id",
        "duration_snapshots",
    }

    missing_columns = required_columns - set(
        state_durations.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if state_durations.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "asset_count",
                "state_frequency",
                "total_snapshots",
                "mean_duration",
                "median_duration",
                "max_duration",
            ]
        )

    summary = (
        state_durations.groupby(
            "cluster_id",
            sort=True,
        )
        .agg(
            asset_count=("symbol", "nunique"),
            state_frequency=("duration_snapshots", "size"),
            total_snapshots=("duration_snapshots", "sum"),
            mean_duration=("duration_snapshots", "mean"),
            median_duration=("duration_snapshots", "median"),
            max_duration=("duration_snapshots", "max"),
        )
        .reset_index()
    )

    return summary






def analyze_state_transitions(
behavioral_history, expected_interval: str | pd.Timedelta = "7D", cluster_ids: list[int] | None = None) -> StateTransitionResult:
    asset_state_paths = build_asset_state_paths(
        behavioral_history=behavioral_history,
        expected_interval=expected_interval,
    )

    transition_events = extract_transition_events(
        asset_state_paths=asset_state_paths,
    )

    if cluster_ids is None:
        cluster_ids = sorted(
            behavioral_history["cluster_id"].unique().tolist()
        )

    market_counts, market_probabilities = (
        build_transition_matrices(
            transition_events=transition_events,
            cluster_ids=cluster_ids,
        )
    )

    asset_matrices = build_asset_transition_matrices(
        transition_events=transition_events,
        cluster_ids=cluster_ids,
    )

    state_durations = calculate_state_duration(
        asset_state_paths=asset_state_paths,
    )

    asset_summary = build_asset_state_summary(
        state_durations=state_durations,
    )

    market_summary = build_market_state_summary(
        state_durations=state_durations,
    )

    return StateTransitionResult(
        asset_state_paths=asset_state_paths,
        transition_events=transition_events,
        market_transition_counts=market_counts,
        market_transition_probabilities=market_probabilities,
        asset_transition_matrices=asset_matrices,
        state_durations=state_durations,
        asset_state_summary=asset_summary,
        market_state_summary=market_summary,
    )
