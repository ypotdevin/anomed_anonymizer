import time
from pathlib import Path
from unittest.mock import MagicMock

import anomed_utils as utils
import falcon
import numpy as np
import pytest
import requests
from falcon import testing

import anomed_anonymizer as anonymizer


@pytest.fixture()
def example_features() -> np.ndarray:
    return np.arange(10)


@pytest.fixture()
def example_targets() -> np.ndarray:
    return np.zeros(shape=(10,))


@pytest.fixture()
def example_dataset(example_features, example_targets) -> dict[str, np.ndarray]:
    return dict(X=example_features, y=example_targets)


class Dummy:
    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(shape=(len(X),))

    def save(self, filepath: str | Path) -> None:
        with open(filepath, "w") as file:
            file.write("test")

    def validate_input(self, X: np.ndarray) -> None:
        if X.dtype != np.int_:
            raise ValueError()


class LongFitDummy(Dummy):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        time.sleep(1)


@pytest.fixture()
def dummy_anonymizer() -> anonymizer.SupervisedLearningAnonymizer:
    return anonymizer.WrappedAnonymizer(Dummy())


@pytest.fixture()
def dummy_long_fit_anonymizer() -> anonymizer.SupervisedLearningAnonymizer:
    return anonymizer.WrappedAnonymizer(LongFitDummy())


@pytest.fixture()
def dummy_server_args(tmp_path):
    return dict(
        anonymizer_identifier="example_anonymizer",
        model_filepath=tmp_path / "model",
        default_batch_size=64,
        training_data_url="http://example.com/train",
        tuning_data_url="http://example.com/tuning",
        validation_data_url="http://example.com/validation",
        utility_evaluation_url="http://example.com/utility",
    )


@pytest.fixture()
def client(dummy_anonymizer, dummy_server_args):
    return testing.TestClient(
        app=anonymizer.supervised_learning_anonymizer_server_factory(
            anonymizer_obj=dummy_anonymizer,
            model_loader=lambda _: dummy_anonymizer,
            **dummy_server_args,
        )
    )


# @pytest.fixture()
# def long_fit_client(dummy_long_fit_anonymizer, dummy_server_args):
#     return testing.TestClient(
#         app=anonymizer.supervised_learning_anonymizer_server_factory(
#             anonymizer_obj=dummy_long_fit_anonymizer,
#             model_loader=lambda _: dummy_long_fit_anonymizer,
#             **dummy_server_args,
#         )
#     )


def test_availability(client):
    message = {"message": "Anonymizer server is alive!"}
    response = client.simulate_get("/")
    assert response.json == message


def test_successful_fit_invocation(client, mocker, example_dataset, dummy_server_args):
    mock = _mock_get_numpy_arrays(mocker, example_dataset)

    response = client.simulate_post("/fit")
    assert mock.call_args.kwargs["url"] == dummy_server_args["training_data_url"]
    assert response.status == falcon.HTTP_CREATED
    assert response.json == dict(message="Fitting has been completed successfully.")


def _mock_get_numpy_arrays(
    _mocker, named_arrays: dict[str, np.ndarray], status_code: int = 200
) -> MagicMock:
    mock_response = _mocker.MagicMock()
    mock_response.status_code = status_code
    mock_response.content = utils.named_ndarrays_to_bytes(named_arrays)
    return _mocker.patch("requests.get", return_value=mock_response)


def test_failing_fit_invocation_data(client, mocker, example_features, example_dataset):
    mock = _mock_get_connection_error(mocker)
    response = client.simulate_post("/fit", params=dict(batch_size=8))
    mock.assert_called_once()
    assert response.status == falcon.HTTP_SERVICE_UNAVAILABLE

    mock = _mock_get_numpy_arrays(mocker, named_arrays=dict(not_X=example_features))
    response = client.simulate_post("/fit", params=dict(batch_size=8))
    mock.assert_called_once()
    assert response.status == falcon.HTTP_INTERNAL_SERVER_ERROR

    malformed_dataset = example_dataset.copy()
    malformed_dataset["X"] = np.ones(shape=(10,), dtype=np.float_)
    mock = _mock_get_numpy_arrays(mocker, named_arrays=malformed_dataset)
    response = client.simulate_post("/fit", params=dict(batch_size=8))
    mock.assert_called_once()
    assert response.status == falcon.HTTP_INTERNAL_SERVER_ERROR


def _mock_get_connection_error(_mocker) -> MagicMock:
    mock_response = _mocker.MagicMock()
    mock_response.side_effect = requests.ConnectionError()
    return _mocker.patch("requests.get", return_value=mock_response)


# def test_failing_fit_invocation_parallel_fit(long_fit_client, mocker, example_dataset):
#     mock_response = mocker.MagicMock()
#     mock_response.status_code = 200
#     mock_response.content = utils.named_ndarrays_to_bytes(example_dataset)
#     mocker.patch("requests.get", return_value=mock_response)

#     _ = long_fit_client.simulate_post("/fit", params=dict(batch_size=8))
#     response = long_fit_client.simulate_post("/fit", params=dict(batch_size=8))
#     assert response.status == falcon.HTTP_SERVICE_UNAVAILABLE


def test_successful_utility_evaluation(
    client, mocker, example_dataset, example_features, dummy_server_args
):
    # Initiate training to have something to evaluate
    _invoke_fit(client, mocker, example_dataset)

    data_mock = _mock_get_numpy_arrays(mocker, dict(X=example_features))
    mock_json_utility_data = {
        "mae": 13.37,
        "rmse": 4.20,
        "coeff_determ": 0.82,
    }
    json_mock = _mock_post_json(mocker, mock_json_utility_data)
    for data_split in ["tuning", "validation"]:
        response = client.simulate_post("/evaluate", params=dict(data_split=data_split))
        assert (
            data_mock.call_args.kwargs["url"]
            == dummy_server_args[
                "tuning_data_url" if data_split == "tuning" else "validation_data_url"
            ]
        )
        assert json_mock.call_args.kwargs["params"] == dict(
            anonymizer=dummy_server_args["anonymizer_identifier"], data_split=data_split
        )

        assert response.status == falcon.HTTP_CREATED
        assert response.json == dict(
            message=f"The anonymizer has been evaluated based on {data_split} data.",
            evaluation=mock_json_utility_data,
        )


def test_failing_utility_evaluation_no_fit(client):
    response = client.simulate_post("/evaluate", params=dict(data_split="tuning"))
    assert response.status == falcon.HTTP_SERVICE_UNAVAILABLE


def _invoke_fit(_client, _mocker, _example_dataset) -> None:
    training_data_mock = _mock_get_numpy_arrays(_mocker, _example_dataset)
    _client.simulate_post("/fit")
    training_data_mock.assert_called_once()


def test_failing_utility_evaluation_data(client, mocker, example_dataset):
    # Initiate training to have something to evaluate
    _invoke_fit(client, mocker, example_dataset)

    exception_mock = _mock_get_connection_error(mocker)
    for i, data_split in enumerate(["tuning", "validation"]):
        response = client.simulate_post("/evaluate", params=dict(data_split=data_split))
        assert response.status == falcon.HTTP_SERVICE_UNAVAILABLE
        assert exception_mock.call_count == i + 1


def test_failing_utility_evaluation_submission(
    client, mocker, example_dataset, example_features
):
    # Initiate training to have something to evaluate
    _invoke_fit(client, mocker, example_dataset)

    data_mock = _mock_get_numpy_arrays(mocker, dict(X=example_features))
    mock_response = mocker.MagicMock()
    mock_response.status_code = 500
    post_mock = mocker.patch("requests.post", return_value=mock_response)
    response = client.simulate_post("/evaluate", params=dict(data_split="validation"))

    assert response.status == falcon.HTTP_INTERNAL_SERVER_ERROR
    data_mock.assert_called_once()
    post_mock.assert_called_once()


def test_failing_utility_evaluation_bad_request(client, mocker, example_dataset):
    _invoke_fit(client, mocker, example_dataset)
    response = client.simulate_post("/evaluate")
    assert response.status == falcon.HTTP_BAD_REQUEST
    response = client.simulate_post("/evaluate", params=dict(data_split="fail"))
    assert response.status == falcon.HTTP_BAD_REQUEST


def _mock_post_json(_mocker, _json, status_code: int = 201) -> MagicMock:
    mock_response = _mocker.MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = _json
    return _mocker.patch("requests.post", return_value=mock_response)


def test_successful_predict(
    client, mocker, dummy_anonymizer, example_dataset, example_features
):
    _invoke_fit(client, mocker, example_dataset)
    batch_size = 8
    response = client.simulate_post(
        "/predict",
        params=dict(batch_size=batch_size),
        body=utils.named_ndarrays_to_bytes(dict(X=example_features)),
    )
    assert response.status == falcon.HTTP_CREATED
    prediction = utils.bytes_to_named_ndarrays(response.content)
    assert np.array_equal(
        prediction["prediction"],
        dummy_anonymizer.predict(example_features, batch_size=batch_size),
    )


def test_failing_predict_bad_request(client, mocker, example_dataset, example_features):
    _invoke_fit(client, mocker, example_dataset)
    response = client.simulate_post(
        "/predict",
        body=b"fail",
    )
    assert response.status == falcon.HTTP_BAD_REQUEST
    response = client.simulate_post(
        "/predict",
        body=utils.named_ndarrays_to_bytes(dict(not_X=example_features)),
    )
    assert response.status == falcon.HTTP_BAD_REQUEST
    response = client.simulate_post(
        "/predict",
        body=utils.named_ndarrays_to_bytes(
            dict(X=np.ones(shape=(10,), dtype=np.float_))
        ),
    )
    assert response.status == falcon.HTTP_BAD_REQUEST
