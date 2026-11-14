
import json

import pandas as pd

from intelligent_portfolio_construction_engine.forecasting.dataset_analyzer import (
    DatasetAnalyzer,
)
from intelligent_portfolio_construction_engine.forecasting.dataset_builder import (
    FEATURE_COLUMNS,
)
from intelligent_portfolio_construction_engine.forecasting.reporting.dataset_reporter import (
    DatasetReporter,
)


def create_dataset(sample_count: int = 40) -> pd.DataFrame:
    records = []

    for index in range(sample_count):
        record = {
            "symbol": f"ASSET_{index % 5}",
            "snapshot_date": pd.Timestamp("2024-01-01")
            + pd.Timedelta(weeks=index),
            "target_date": pd.Timestamp("2024-01-01")
            + pd.Timedelta(weeks=index + 4),
            "model_version": "model_1",
            "target_model_version": "model_1",
            "cluster_id": index % 3,
            "target_cluster_id": index % 3,
        }

        record.update(
            {
                feature: float(index + 1)
                for feature in FEATURE_COLUMNS
            }
        )

        records.append(record)

    return pd.DataFrame(records)


def test_generate_creates_expected_artifacts(tmp_path) -> None:
    dataset = create_dataset()

    analysis = DatasetAnalyzer().analyze(dataset)

    reporter = DatasetReporter(
        output_dir=tmp_path,
    )

    run_path = reporter.generate(
        analysis=analysis,
        dataset=dataset,
    )

    assert run_path.exists()
    assert run_path.is_dir()

    assert (run_path / "dataset_quality.json").exists()
    assert (run_path / "feature_statistics.csv").exists()
    assert (run_path / "target_distribution.csv").exists()
    assert (run_path / "report.html").exists()

    assert (run_path / "figures").is_dir()


def test_generate_creates_feature_distribution_figure(tmp_path) -> None:
    dataset = create_dataset()

    analysis = DatasetAnalyzer().analyze(dataset)

    reporter = DatasetReporter(
        output_dir=tmp_path,
    )

    run_path = reporter.generate(
        analysis=analysis,
        dataset=dataset,
    )

    figures = list((run_path / "figures").glob("*.png"))

    assert figures


def test_generate_creates_valid_json(tmp_path) -> None:
    dataset = create_dataset()

    analysis = DatasetAnalyzer().analyze(dataset)

    reporter = DatasetReporter(
        output_dir=tmp_path,
    )

    run_path = reporter.generate(
        analysis=analysis,
        dataset=dataset,
    )

    json_path = run_path / "dataset_quality.json"

    with json_path.open(encoding="utf-8") as file:
        content = json.load(file)

    assert isinstance(content, dict)


def test_generate_creates_readable_html_report(tmp_path) -> None:
    dataset = create_dataset()

    analysis = DatasetAnalyzer().analyze(dataset)

    reporter = DatasetReporter(
        output_dir=tmp_path,
    )

    run_path = reporter.generate(
        analysis=analysis,
        dataset=dataset,
    )

    html = (run_path / "report.html").read_text(
        encoding="utf-8",
    )

    assert "<html" in html.lower()
    assert "<table" in html.lower()
    assert "utf-8" in html.lower()
    assert "Dataset" in html


def test_generate_creates_separate_run_directories(tmp_path) -> None:
    dataset = create_dataset()

    analysis = DatasetAnalyzer().analyze(dataset)

    reporter = DatasetReporter(
        output_dir=tmp_path,
    )

    first_run = reporter.generate(analysis, dataset)
    second_run = reporter.generate(analysis, dataset)

    assert first_run != second_run
    assert first_run.exists()
    assert second_run.exists()


def test_generate_does_not_modify_original_dataset(tmp_path) -> None:
    dataset = create_dataset()
    original = dataset.copy(deep=True)

    analysis = DatasetAnalyzer().analyze(dataset)

    reporter = DatasetReporter(
        output_dir=tmp_path,
    )

    reporter.generate(analysis, dataset)

    pd.testing.assert_frame_equal(dataset, original)
