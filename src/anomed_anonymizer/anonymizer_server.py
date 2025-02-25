import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

import anomed_utils as utils
import falcon
import numpy as np
import requests

from . import anonymizer

__all__ = [
    "InferenceResource",
    "supervised_learning_anonymizer_server_factory",
    "validate_anonymizer_input_or_raise",
]

_logger = logging.getLogger(__name__)


class InferenceResource:
    """This resource is intended for providing inference access to a fitted
    anonymizer (that follows the supervised learning paradigm) and to invoke its
    evaluation.
    """

    def __init__(
        self,
        anonymizer_identifier: str,
        model_filepath: str | Path,
        model_loader: Callable[[str | Path], anonymizer.SupervisedLearningAnonymizer],
        default_batch_size: int,
        tuning_data_url: str,
        validation_data_url: str,
        utility_evaluation_url: str,
        download_timeout: float,
        upload_timeout: float,
    ) -> None:
        """
        Parameters
        ----------
        anonymizer_identifier : str
            The identifier of the anonymizer which will be loaded for inference
            or evaluation. This is required by the corresponding challenge's
            evaluation API.
        model_filepath : str | Path
            Where to load a fitted model from for inference (argument to
            `model_loader`).
        model_loader : Callable[[str  |  Path], anonymizer.SupervisedLearningAnonymizer]
            A function that is able to recover an instance of
            `anonymizer.SupervisedLearningAnonymizer` from `model_filepath`. In
            case `anonymizer_obj` is a wrapped object, keep in mind that this
            loader has to recover the outer object too, not just the inner
            object.
        default_batch_size : int
            A default batch size to use, if none is specified in the requests.
        tuning_data_url : str
            Where to obtain tuning data from (usually, this points to an API of
            the challenge you are submitting your anonymizer to).
        validation_data_url : str
            Where to obtain validation data from (usually, this points to an API
            of the challenge you are submitting your anonymizer to).
        utility_evaluation_url : str
            Where to submit the target values this anonymizer inferred from the
            validation data features (usually, this points to an API of the
            challenge you are submitting your anonymizer to).
        download_timeout : float, optional
            The time in seconds to wait before a download connection is
            considered faulty.
        upload_timeout : float, optional
            The time in seconds to wait before an upload connection is
            considered faulty.
        """
        self._anon_id = anonymizer_identifier
        self._model_filepath = Path(model_filepath)
        self._load_model = model_loader
        self._default_batch_size = default_batch_size
        self._url_mapper = dict(
            tuning=tuning_data_url,
            validation=validation_data_url,
            utility=utility_evaluation_url,
        )
        self._download_timeout = download_timeout
        self._upload_timeout = upload_timeout
        self._loaded_model: anonymizer.SupervisedLearningAnonymizer = None  # type: ignore
        self._loaded_model_modification_time: datetime = None  # type: ignore
        self._expected_array_label = "X"

    def on_post_predict(self, req: falcon.Request, resp: falcon.Response) -> None:
        self._load_most_recent_model()

        array_bytes = req.bounded_stream.read()
        array = utils.bytes_to_named_ndarrays_or_raise(
            array_bytes,
            expected_array_labels=[self._expected_array_label],
            error_status=falcon.HTTP_BAD_REQUEST,
            error_message="Supplied array is not compatible with the anonymizer.",
        )
        X = array[self._expected_array_label]
        validate_anonymizer_input_or_raise(
            X,
            self._loaded_model,
            error_status=falcon.HTTP_BAD_REQUEST,
            error_message="Supplied array is not compatible with the anonymizer.",
        )

        batch_size = req.get_param_as_int(
            "batch_size", default=self._default_batch_size
        )
        _logger.debug(
            f"Inferring targets for feature array with shape {X.shape} and "
            f"dtype {X.dtype}. Using batch size {batch_size}"
        )
        prediction = self._loaded_model.predict(X=X, batch_size=batch_size)
        resp.data = utils.named_ndarrays_to_bytes(dict(prediction=prediction))
        resp.status = falcon.HTTP_CREATED

    def on_post_evaluate(self, req: falcon.Request, resp: falcon.Response) -> None:
        self._load_most_recent_model()
        data_split = req.get_param("data_split", required=True)
        if data_split not in self._url_mapper.keys():
            raise falcon.HTTPBadRequest(
                description=f"Invalid value for parameter 'data_split': {data_split}. "
                "It needs to be 'tuning', or 'validation'."
            )
        array = utils.get_named_arrays_or_raise(
            data_url=self._url_mapper[data_split],
            expected_array_labels=[self._expected_array_label],
            timeout=self._download_timeout,
        )

        X = array[self._expected_array_label]
        _logger.debug(
            f"Received feature array for evaluation with shape {X.shape} and dtype "
            f"{X.dtype}. Inferring its targets using batch size "
            f"{self._default_batch_size}"
        )
        prediction = self._loaded_model.predict(X, self._default_batch_size)
        _logger.debug(
            f"Inferred targets for evaluation. The target's shape is {prediction.shape}"
            f" and its dtype is {prediction.dtype}"
        )
        try:
            evaluation_response = requests.post(
                url=self._url_mapper["utility"],
                data=utils.named_ndarrays_to_bytes(dict(prediction=prediction)),
                params=dict(anonymizer=self._anon_id, data_split=data_split),
                timeout=self._upload_timeout,
            )
            if evaluation_response.status_code != 201:
                raise ValueError
            resp.text = json.dumps(
                dict(
                    message=(
                        f"The anonymizer has been evaluated based on {data_split} data."
                    ),
                    evaluation=evaluation_response.json(),
                )
            )
            resp.status = falcon.HTTP_CREATED
        except ValueError:
            raise falcon.HTTPInternalServerError(
                description="Utility evaluation failed."
            )
        except requests.Timeout:
            raise falcon.HTTPServiceUnavailable(
                description="Challenge currently not available for evaluation."
            )

    def _load_most_recent_model(self) -> None:
        if not self._model_filepath.exists():
            raise falcon.HTTPServiceUnavailable(
                description="This anonymizer is not fitted/trained yet.",
            )
        mod_time_from_disk = datetime.fromtimestamp(
            self._model_filepath.stat().st_mtime
        )
        if _is_older(self._loaded_model_modification_time, mod_time_from_disk):
            self._loaded_model = self._load_model(self._model_filepath)
            self._loaded_model_modification_time = mod_time_from_disk
        else:
            # keep the current model as it is already recent enough
            pass


def _is_older(dt1: datetime | None, dt2: datetime) -> bool:
    """Tell whether `dt1` is older (i.e. more in the past) than `dt2`. If `dt1`
    is the same as `dt2`, or even if `dt1` is `None`, output `True`."""
    if dt1 is None:
        return True
    else:
        return dt1 <= dt2


def supervised_learning_anonymizer_server_factory(
    anonymizer_identifier: str,
    anonymizer_obj: anonymizer.SupervisedLearningAnonymizer,
    model_filepath: str | Path,
    default_batch_size: int,
    training_data_url: str,
    tuning_data_url: str,
    validation_data_url: str,
    utility_evaluation_url: str,
    model_loader: Callable[[str | Path], anonymizer.SupervisedLearningAnonymizer],
    download_timeout: float = 10.0,
    upload_timeout: float = 10.0,
) -> falcon.App:
    """A factory to create a web application object which hosts an
    `anonymizer.SupervisedLearningAnonymizer`, currently the most basic use
    case of anonymizers (privacy preserving ML models) for the AnoMed
    competition platform.

    By using this factory, you don't have to worry any web-programming issues,
    as they are hidden from you. The generated web app will feature the
    following routes (more details may be found in this project's openapi
    specification):

    * [GET] `/`
    * [POST] `/fit`
    * [POST] `/evaluate`
    * [POST] `/predict`

    Parameters
    ----------
    anonymizer_identifier : str
        Passed to `InferenceResource`.
    anonymizer_obj : anonymizer.SupervisedLearningAnonymizer
        An anonymizer that is based on the supervised learning paradigm.
    model_filepath : str | Path
        Where to save the fitted model after training / where to load it back
        from for inference.
    default_batch_size : int
        Passed to `InferenceResource`.
    training_data_url : str
        Where to obtain training data from (usually, this points to an API of
        the challenge you are submitting your anonymizer to).
    tuning_data_url : str
        Passed to `InferenceResource`.
    validation_data_url : str
        Passed to `InferenceResource`.
    utility_evaluation_url : str
        Passed to `InferenceResource`.
    model_loader : Callable[[str  |  Path], anonymizer.SupervisedLearningAnonymizer]
        Passed to `InferenceResource`.
    download_timeout : float, optional
        The time in seconds to wait before a download connection is considered
        faulty. By default 10.0. You might want to increase this if you expect
        that it takes some time to download the training data from the
        challenge.
    upload_timeout : float, optional
        The time in seconds to wait before an upload connection is considered
        faulty. By default 10.0. You might want to increase this if you expect
        that it takes some time to upload predictions to the challenge.

    Returns
    -------
    falcon.App
        A web application object based on the falcon web framework.
    """
    app = falcon.App()

    app.add_route(
        "/", utils.StaticJSONResource(dict(message="Anonymizer server is alive!"))
    )
    app.add_route(
        "/fit",
        utils.FitResource(
            data_getter=_get_anonymizer_fit_data(
                anonymizer=anonymizer_obj,
                training_data_url=training_data_url,
                timeout=download_timeout,
            ),
            model=anonymizer_obj,
            model_filepath=model_filepath,
        ),
    )
    ir = InferenceResource(
        anonymizer_identifier=anonymizer_identifier,
        model_filepath=model_filepath,
        model_loader=model_loader,
        default_batch_size=default_batch_size,
        tuning_data_url=tuning_data_url,
        validation_data_url=validation_data_url,
        utility_evaluation_url=utility_evaluation_url,
        download_timeout=download_timeout,
        upload_timeout=upload_timeout,
    )
    app.add_route("/evaluate", ir, suffix="evaluate")
    app.add_route("/predict", ir, suffix="predict")
    return app


def validate_anonymizer_input_or_raise(
    feature_array: np.ndarray,
    anonymizer: anonymizer.SupervisedLearningAnonymizer,
    error_status: str | int | None = falcon.HTTP_INTERNAL_SERVER_ERROR,
    error_message: str | None = None,
) -> None:
    """Validate the input for an anonymizer. If validation fails, raise a
    `falcon.HTTPError` instead.

    Parameters
    ----------
    feature_array : np.ndarray
        A NumPy array containing the features for this anonymizer.
    anonymizer : anonymizer.SupervisedLearningAnonymizer
        The anonymizer to validate input for. This function will use the
        anonymizer's `validate_input` method.
    error_status : str | int | None, optional
        The error status to use if validation fails. By default,
        `falcon.HTTP_INTERNAL_SERVER_ERROR`.
    error_message : str | None, optional
        The error message to output. By default `None`, which will result in a
        generic message derived from the `error_status`.

    Raises
    ------
    falcon.HTTPError
        If validation fails.
    """
    try:
        anonymizer.validate_input(feature_array)
    except ValueError:
        if error_status is None:
            error_status = falcon.HTTP_INTERNAL_SERVER_ERROR
        raise falcon.HTTPError(status=error_status, description=error_message)


def _get_anonymizer_fit_data(
    anonymizer: anonymizer.SupervisedLearningAnonymizer,
    training_data_url: str,
    timeout: float,
    expected_array_labels: list[str] | None = None,
) -> Callable[[], dict[str, np.ndarray]]:
    if expected_array_labels is None:
        expected_array_labels = ["X", "y"]

    def getter():
        training_data = utils.get_named_arrays_or_raise(
            data_url=training_data_url,
            expected_array_labels=expected_array_labels,
            timeout=timeout,
        )

        validate_anonymizer_input_or_raise(
            training_data[expected_array_labels[0]],
            anonymizer,
            falcon.HTTP_INTERNAL_SERVER_ERROR,
            "The anonymizer is not compatible with the training data.",
        )
        return training_data

    return getter
