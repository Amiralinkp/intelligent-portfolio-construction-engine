
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from intelligent_portfolio_construction_engine.forecasting.dataset_builder import (
    FEATURE_COLUMNS,
)
from intelligent_portfolio_construction_engine.models.forecasting_dataset_analysis import (
    ForecastingDatasetAnalysis,
)


class DatasetReporter:
    def __init__(
        self,
        output_dir: str | Path = "artifacts/forecasting/dataset_analysis",
    ) -> None:
        self.output_dir = Path(output_dir)

    def generate(
        self,
        analysis: ForecastingDatasetAnalysis,
        dataset: pd.DataFrame,
    ) -> Path:
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            + f"_{uuid4().hex[:8]}"
        )

        run_dir = self.output_dir / f"run_{run_id}"
        figures_dir = run_dir / "figures"

        figures_dir.mkdir(parents=True, exist_ok=False)

        self._save_json(
            {
                "run_id": run_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset_summary": analysis.dataset_summary,
                "quality_report": analysis.quality_report,
                "warnings": analysis.warnings,
                "recommendations": analysis.recommendations,
            },
            run_dir / "dataset_quality.json",
        )

        analysis.feature_statistics.to_csv(
            run_dir / "feature_statistics.csv",
            index=False,
        )

        analysis.target_distribution.to_csv(
            run_dir / "target_distribution.csv",
            index=False,
        )

        self._plot_target_distribution(
            analysis.target_distribution,
            figures_dir / "target_distribution.png",
        )

        self._plot_feature_distributions(
            dataset,
            figures_dir / "feature_distributions.png",
        )

        self._save_html_report(
            analysis=analysis,
            run_id=run_id,
            output_path=run_dir / "report.html",
        )

        return run_dir

    @staticmethod
    def _save_json(
        data: dict,
        output_path: Path,
    ) -> None:
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False,
                default=DatasetReporter._json_default,
            )

    @staticmethod
    def _json_default(value: object) -> object:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()

        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable."
        )

    @staticmethod
    def _plot_target_distribution(
        distribution: pd.DataFrame,
        output_path: Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))

        if distribution.empty:
            ax.text(
                0.5,
                0.5,
                "No valid target labels available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        else:
            labels = distribution["cluster_id"].astype(str)

            ax.bar(
                labels,
                distribution["sample_count"],
            )

            ax.set_title("Forecast Target Cluster Distribution")
            ax.set_xlabel("Target Cluster")
            ax.set_ylabel("Sample Count")

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    
    @staticmethod
    def _plot_feature_distributions(
        dataset: pd.DataFrame,
        output_path: Path,
    ) -> None:
        columns = 3
        rows = max(1, int(np.ceil(len(FEATURE_COLUMNS) / columns)))

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(15, 4 * rows),
            squeeze=False,
        )

        for index, feature in enumerate(FEATURE_COLUMNS):
            row, column = divmod(index, columns)
            ax = axes[row][column]

            values = pd.to_numeric(
                dataset[feature],
                errors="coerce",
            ).to_numpy(
                dtype=float,
                na_value=np.nan,
            )

            values = values[np.isfinite(values)]

            if len(values) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No valid values",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue

            ax.hist(values, bins="auto")
            ax.set_title(feature)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

        for index in range(len(FEATURE_COLUMNS), rows * columns):
            row, column = divmod(index, columns)
            axes[row][column].set_axis_off()

        fig.suptitle("Feature Distributions")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


    @staticmethod
    def _save_html_report(
        analysis: ForecastingDatasetAnalysis,
        run_id: str,
        output_path: Path,
    ) -> None:
        summary = analysis.dataset_summary
        quality = analysis.quality_report

        summary_table = pd.DataFrame(
            {
                "Metric": list(summary.keys()),
                "Value": list(summary.values()),
            }
        ).to_html(index=False, escape=True)

        quality_table = pd.DataFrame(
            {
                "Metric": list(quality.keys()),
                "Value": [
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        default=DatasetReporter._json_default,
                    )
                    if isinstance(value, (dict, list))
                    else value
                    for value in quality.values()
                ],
            }
        ).to_html(index=False, escape=True)

        feature_table = analysis.feature_statistics.to_html(
            index=False,
            float_format=lambda value: f"{value:.4f}",
            escape=True,
        )

        target_table = analysis.target_distribution.to_html(
            index=False,
            float_format=lambda value: f"{value:.4f}",
            escape=True,
        )

        warnings_html = (
            "".join(
                f"<li>{escape(warning)}</li>"
                for warning in analysis.warnings
            )
            or "<li>No warnings detected.</li>"
        )

        recommendations_html = (
            "".join(
                f"<li>{escape(item)}</li>"
                for item in analysis.recommendations
            )
            or "<li>No additional recommendations.</li>"
        )

        warning_count = len(analysis.warnings)

        html_report = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forecasting Dataset Analysis</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 0 24px;
            line-height: 1.6;
            color: #222;
            background-color: #fff;
        }}
        h1, h2 {{
            margin-top: 32px;
        }}
        h1 {{
            border-bottom: 2px solid #ddd;
            padding-bottom: 12px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0 28px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
            overflow-wrap: anywhere;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 16px 0 28px;
        }}
        .metadata {{
            color: #666;
            font-size: 0.9rem;
        }}
        .summary {{
            padding: 16px;
            background-color: #f5f5f5;
            border-radius: 6px;
        }}
    </style>
</head>
<body>
    <h1>Forecasting Dataset Analysis</h1>

    <p class="metadata">Run ID: {escape(run_id)}</p>

    <h2>1. Executive Summary</h2>
    <div class="summary">
        <p>
            The dataset contains {summary.get("sample_count", 0)} samples
            across {summary.get("asset_count", 0)} assets.
        </p>
        <p>
            The analysis identified {warning_count} warning(s).
        </p>
    </div>

    <h2>2. Dataset Overview</h2>
    {summary_table}

    <h2>3. Data Quality Assessment</h2>
    {quality_table}

    <h2>4. Feature Analysis</h2>
    {feature_table}

    <img
        src="figures/feature_distributions.png"
        alt="Feature distribution overview"
    >

    <h2>5. Target Distribution</h2>
    {target_table}

    <img
        src="figures/target_distribution.png"
        alt="Target cluster distribution"
    >

    <h2>6. Warnings</h2>
    <ul>{warnings_html}</ul>

    <h2>7. Recommendations</h2>
    <ul>{recommendations_html}</ul>
</body>
</html>
"""

        output_path.write_text(
            html_report,
            encoding="utf-8",
        )
