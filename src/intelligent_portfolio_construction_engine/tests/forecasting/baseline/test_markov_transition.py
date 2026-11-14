
import numpy as np
import pandas as pd
import pytest

from intelligent_portfolio_construction_engine.forecasting.baselines.markov_transition import (
    MarkovTransitionBaseline,
)


@pytest.fixture
def training_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster_id": [1, 1, 1, 2, 2],
            "target_cluster_id": [2, 2, 3, 3, 3],
        }
    )


def test_fit_returns_fitted_model(
    training_dataset: pd.DataFrame,
) -> None:
    model = MarkovTransitionBaseline()

    result = model.fit(training_dataset)

    assert result is model
    assert model._is_fitted is True


def test_predict_selects_most_likely_destination(
    training_dataset: pd.DataFrame,
) -> None:
    model = MarkovTransitionBaseline()
    model.fit(training_dataset)

    dataset = pd.DataFrame(
        {"cluster_id": [1, 2]},
        index=[10, 20],
    )

    predictions = model.predict(dataset)

    expected = pd.Series(
        [2, 3],
        index=[10, 20],
        name="predicted_cluster_id",
    )

    pd.testing.assert_series_equal(
        predictions,
        expected,
    )


def test_transition_matrix_uses_laplace_smoothing(
    training_dataset: pd.DataFrame,
) -> None:
    model = MarkovTransitionBaseline(alpha=1.0)
    model.fit(training_dataset)

    matrix = model._transition_matrix

    assert (matrix > 0).all().all()

    row_sums = matrix.sum(axis=1)

    assert np.allclose(row_sums.to_numpy(), 1.0)


def test_unseen_source_state_uses_persistence_fallback(
    training_dataset: pd.DataFrame,
) -> None:
    model = MarkovTransitionBaseline()
    model.fit(training_dataset)

    dataset = pd.DataFrame(
        {"cluster_id": [99]}
    )

    predictions = model.predict(dataset)

    assert predictions.iloc[0] == 99


def test_predict_requires_fitted_model() -> None:
    model = MarkovTransitionBaseline()

    dataset = pd.DataFrame(
        {"cluster_id": [1]}
    )

    with pytest.raises(RuntimeError):
        model.predict(dataset)


def test_fit_rejects_empty_dataset() -> None:
    model = MarkovTransitionBaseline()

    empty_dataset = pd.DataFrame(
        columns=["cluster_id", "target_cluster_id"]
    )

    with pytest.raises(ValueError):
        model.fit(empty_dataset)


@pytest.mark.parametrize(
    "alpha",
    [
        0,
        -1,
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_invalid_alpha_is_rejected(
    alpha: float,
) -> None:
    with pytest.raises(ValueError):
        MarkovTransitionBaseline(alpha=alpha)


def test_fit_rejects_missing_target_column() -> None:
    model = MarkovTransitionBaseline()

    dataset = pd.DataFrame(
        {"cluster_id": [1, 2]}
    )

    with pytest.raises(ValueError):
        model.fit(dataset)


def test_fit_rejects_missing_cluster_ids() -> None:
    model = MarkovTransitionBaseline()

    dataset = pd.DataFrame(
        {
            "cluster_id": [1, None],
            "target_cluster_id": [2, 3],
        }
    )

    with pytest.raises(ValueError):
        model.fit(dataset)
