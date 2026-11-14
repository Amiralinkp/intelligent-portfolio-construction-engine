
import numpy as np
import pandas as pd
import pytest

from intelligent_portfolio_construction_engine.forecasting.evaluation.classification_evaluator import (
    ClassificationEvaluator,
)


@pytest.fixture
def evaluator() -> ClassificationEvaluator:
    return ClassificationEvaluator()


def test_evaluate_returns_expected_metrics(
    evaluator: ClassificationEvaluator,
) -> None:
    actual = [1, 1, 2, 2]
    predicted = [1, 2, 2, 2]

    result = evaluator.evaluate(actual, predicted)

    assert result.overall_metrics["sample_count"] == 4
    assert result.overall_metrics["accuracy"] == 0.75
    assert result.overall_metrics["balanced_accuracy"] == 0.75
    assert result.overall_metrics["macro_f1"] == pytest.approx(
        0.7333333333
    )


def test_confusion_matrix_contains_correct_counts(
    evaluator: ClassificationEvaluator,
) -> None:
    actual = [1, 1, 2, 2]
    predicted = [1, 2, 2, 2]

    result = evaluator.evaluate(actual, predicted)

    expected = pd.DataFrame(
        [[1, 1], [0, 2]],
        index=pd.Index([1, 2], name="actual_cluster"),
        columns=pd.Index([1, 2], name="predicted_cluster"),
    )

    pd.testing.assert_frame_equal(
        result.confusion_matrix,
        expected,
    )


def test_class_metrics_include_precision_recall_f1(
    evaluator: ClassificationEvaluator,
) -> None:
    result = evaluator.evaluate(
        [1, 1, 2, 2],
        [1, 2, 2, 2],
    )

    assert {
        "precision",
        "recall",
        "f1_score",
        "support",
    }.issubset(result.class_metrics.columns)

    assert result.class_metrics.loc[1, "support"] == 2
    assert result.class_metrics.loc[2, "support"] == 2


def test_known_labels_include_unseen_test_states(
    evaluator: ClassificationEvaluator,
) -> None:
    result = evaluator.evaluate(
        y_true=[1, 2, 3],
        y_pred=[1, 2, 2],
        labels=[1, 2],
        known_labels=[1, 2],
    )

    assert 3 in result.confusion_matrix.index
    assert result.error_analysis["unseen_state_count"] == 1
    assert result.error_analysis["unseen_state_rate"] == pytest.approx(
        1 / 3
    )


def test_unseen_predictions_are_counted(
    evaluator: ClassificationEvaluator,
) -> None:
    result = evaluator.evaluate(
        y_true=[1, 2],
        y_pred=[1, 99],
        known_labels=[1, 2],
    )

    assert result.error_analysis["unseen_prediction_count"] == 1


def test_error_analysis_counts_correct_and_incorrect_predictions(
    evaluator: ClassificationEvaluator,
) -> None:
    result = evaluator.evaluate(
        y_true=[1, 1, 2, 2],
        y_pred=[1, 2, 2, 3],
    )

    assert result.error_analysis["correct_count"] == 2
    assert result.error_analysis["incorrect_count"] == 2
    assert result.error_analysis["error_rate"] == 0.5


def test_evaluate_rejects_unequal_lengths(
    evaluator: ClassificationEvaluator,
) -> None:
    with pytest.raises(ValueError):
        evaluator.evaluate(
            [1, 2, 3],
            [1, 2],
        )


def test_evaluate_rejects_empty_data(
    evaluator: ClassificationEvaluator,
) -> None:
    with pytest.raises(ValueError):
        evaluator.evaluate([], [])


@pytest.mark.parametrize(
    "actual,predicted",
    [
        ([1, None], [1, 2]),
        ([1, 2], [1, np.nan]),
    ],
)
def test_evaluate_rejects_missing_values(
    evaluator: ClassificationEvaluator,
    actual: list,
    predicted: list,
) -> None:
    with pytest.raises(ValueError):
        evaluator.evaluate(actual, predicted)


def test_macro_f1_is_not_accuracy(
    evaluator: ClassificationEvaluator,
) -> None:
    result = evaluator.evaluate(
        [1, 1, 1, 2],
        [1, 1, 2, 2],
    )

    assert result.overall_metrics["accuracy"] == 0.75
    assert result.overall_metrics["macro_f1"] != 0.75


def test_unseen_state_analysis_can_be_disabled(
    evaluator: ClassificationEvaluator,
) -> None:
    result = evaluator.evaluate(
        [1, 2],
        [1, 2],
    )

    assert (
        result.error_analysis["unseen_state_analysis_enabled"] == 0
    )
