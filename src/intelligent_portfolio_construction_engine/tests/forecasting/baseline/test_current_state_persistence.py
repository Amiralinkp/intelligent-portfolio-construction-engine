
import pandas as pd
import pytest

from intelligent_portfolio_construction_engine.forecasting.baselines.current_state_persistence import (
    CurrentStatePersistenceBaseline,
)


@pytest.fixture
def training_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cluster_id": [1, 2, 3],
            "target_cluster_id": [2, 2, 1],
        }
    )


@pytest.fixture
def prediction_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {"cluster_id": [1, 3, 2]},
        index=[10, 20, 30],
    )


def test_fit_returns_fitted_model(
    training_dataset: pd.DataFrame,
) -> None:
    model = CurrentStatePersistenceBaseline()

    result = model.fit(training_dataset)

    assert result is model
    assert model._is_fitted is True


def test_predict_returns_current_clusters(
    training_dataset: pd.DataFrame,
    prediction_dataset: pd.DataFrame,
) -> None:
    model = CurrentStatePersistenceBaseline()
    model.fit(training_dataset)

    predictions = model.predict(prediction_dataset)

    expected = pd.Series(
        [1, 3, 2],
        index=[10, 20, 30],
        name="predicted_cluster_id",
    )

    pd.testing.assert_series_equal(
        predictions,
        expected,
    )


def test_predict_requires_fitted_model(
    prediction_dataset: pd.DataFrame,
) -> None:
    model = CurrentStatePersistenceBaseline()

    with pytest.raises(RuntimeError):
        model.predict(prediction_dataset)


def test_fit_rejects_empty_dataset() -> None:
    model = CurrentStatePersistenceBaseline()

    with pytest.raises(ValueError):
        model.fit(pd.DataFrame(columns=["cluster_id"]))


def test_predict_rejects_missing_cluster_column() -> None:
    model = CurrentStatePersistenceBaseline()
    model.fit(pd.DataFrame({"cluster_id": [1]}))

    with pytest.raises(ValueError):
        model.predict(pd.DataFrame({"symbol": ["AAPL"]}))


def test_predict_rejects_missing_cluster_ids() -> None:
    model = CurrentStatePersistenceBaseline()
    model.fit(pd.DataFrame({"cluster_id": [1]}))

    invalid_dataset = pd.DataFrame(
        {"cluster_id": [1, None]}
    )

    with pytest.raises(ValueError):
        model.predict(invalid_dataset)
