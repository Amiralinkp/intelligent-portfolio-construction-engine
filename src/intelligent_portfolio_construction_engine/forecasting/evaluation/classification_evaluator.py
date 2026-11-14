
from collections.abc import Sequence

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from intelligent_portfolio_construction_engine.models.forecasting_evaluation_result import (
    ForecastingEvaluationResult,
)


class ClassificationEvaluator:
    def evaluate(
        self,
        y_true: Sequence[object] | pd.Series,
        y_pred: Sequence[object] | pd.Series,
        labels: Sequence[object] | None = None,
        known_labels: Sequence[object] | None = None,
    ) -> ForecastingEvaluationResult:
        actual = self._validate_values(y_true, "y_true")
        predicted = self._validate_values(y_pred, "y_pred")

        if len(actual) != len(predicted):
            raise ValueError(
                "Actual and predicted values must have equal lengths."
            )

        if actual.empty:
            raise ValueError(
                "Evaluation data cannot be empty."
            )

        observed_labels = self._resolve_labels(
            actual,
            predicted,
            None,
        )

        report_labels = self._resolve_labels(
            actual,
            predicted,
            labels,
        )

        overall_metrics = self._calculate_overall_metrics(
            actual,
            predicted,
            observed_labels,
        )

        class_metrics = self._calculate_class_metrics(
            actual,
            predicted,
            report_labels,
        )

        matrix = self._calculate_confusion_matrix(
            actual,
            predicted,
            report_labels,
        )

        error_analysis = self._analyze_errors(
            actual,
            predicted,
            known_labels,
        )

        return ForecastingEvaluationResult(
            overall_metrics=overall_metrics,
            class_metrics=class_metrics,
            confusion_matrix=matrix,
            error_analysis=error_analysis,
        )

    @staticmethod
    def _validate_values(
        values: Sequence[object] | pd.Series,
        name: str,
    ) -> pd.Series:
        result = pd.Series(values).reset_index(drop=True)

        if result.isna().any():
            raise ValueError(
                f"{name} cannot contain missing values."
            )

        return result

    @staticmethod
    def _resolve_labels(
        actual: pd.Series,
        predicted: pd.Series,
        labels: Sequence[object] | None,
    ) -> list[object]:
        observed_labels = pd.unique(
            pd.concat(
                [actual, predicted],
                ignore_index=True,
            )
        ).tolist()

        if labels is None:
            return observed_labels

        resolved_labels = list(dict.fromkeys(labels))

        for label in observed_labels:
            if label not in resolved_labels:
                resolved_labels.append(label)

        return resolved_labels

    @staticmethod
    def _calculate_overall_metrics(
        actual: pd.Series,
        predicted: pd.Series,
        labels: list[object],
    ) -> dict[str, float | int]:
        macro_scores = precision_recall_fscore_support(
            actual,
            predicted,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        weighted_scores = precision_recall_fscore_support(
            actual,
            predicted,
            labels=labels,
            average="weighted",
            zero_division=0,
        )

        return {
            "sample_count": len(actual),
            "accuracy": float(
                accuracy_score(actual, predicted)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(actual, predicted)
            ),
            "macro_precision": float(macro_scores[0]),
            "macro_recall": float(macro_scores[1]),
            "macro_f1": float(macro_scores[2]),
            "weighted_precision": float(weighted_scores[0]),
            "weighted_recall": float(weighted_scores[1]),
            "weighted_f1": float(weighted_scores[2]),
        }

    @staticmethod
    def _calculate_class_metrics(
        actual: pd.Series,
        predicted: pd.Series,
        labels: list[object],
    ) -> pd.DataFrame:
        precision, recall, f1, support = (
            precision_recall_fscore_support(
                actual,
                predicted,
                labels=labels,
                zero_division=0,
            )
        )

        return pd.DataFrame(
            {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": support,
            },
            index=pd.Index(labels, name="cluster_id"),
        )

    @staticmethod
    def _calculate_confusion_matrix(
        actual: pd.Series,
        predicted: pd.Series,
        labels: list[object],
    ) -> pd.DataFrame:
        matrix = confusion_matrix(
            actual,
            predicted,
            labels=labels,
        )

        return pd.DataFrame(
            matrix,
            index=pd.Index(labels, name="actual_cluster"),
            columns=pd.Index(labels, name="predicted_cluster"),
        )

    @staticmethod
    def _analyze_errors(
        actual: pd.Series,
        predicted: pd.Series,
        known_labels: Sequence[object] | None,
    ) -> dict[str, int | float]:
        correct = actual.eq(predicted)
        incorrect_count = int((~correct).sum())

        if known_labels is None:
            unseen_state_count = 0
            unseen_state_analysis_enabled = 0
            unseen_prediction_count = 0
        else:
            known_states = set(known_labels)

            unseen_state_count = int(
                (~actual.isin(known_states)).sum()
            )

            unseen_prediction_count = int(
                (~predicted.isin(known_states)).sum()
            )

            unseen_state_analysis_enabled = 1

        sample_count = len(actual)

        return {
            "sample_count": sample_count,
            "correct_count": int(correct.sum()),
            "incorrect_count": incorrect_count,
            "error_rate": incorrect_count / sample_count,
            "unseen_state_count": unseen_state_count,
            "unseen_state_rate": unseen_state_count / sample_count,
            "unseen_prediction_count": unseen_prediction_count,
            "unseen_state_analysis_enabled": (
                unseen_state_analysis_enabled
            ),
        }
