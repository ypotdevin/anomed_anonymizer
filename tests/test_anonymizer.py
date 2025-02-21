from pathlib import Path

import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB

from anomed_anonymizer import anonymizer


@pytest.fixture()
def empty_ndarray() -> np.ndarray:
    return np.array([])


@pytest.fixture()
def ten_elem_ndarray() -> np.ndarray:
    return np.arange(10)


@pytest.fixture()
def ten_ones_ndarray() -> np.ndarray:
    return np.ones(shape=(10,))


class OnlyFitDummy:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass


class OnlyPredictDummy:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(shape=(len(X),))


class PartialDummy(OnlyFitDummy, OnlyPredictDummy):
    pass


class CompleteDummy:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    # note the presence of the batch_size argument
    def predict(self, X: np.ndarray, batch_size: int) -> np.ndarray:
        return np.ones(shape=(len(X),))

    def save(self, filepath: str | Path) -> None:
        anonymizer.pickle_anonymizer(self, filepath)

    def validate_input(self, X: np.ndarray) -> None:
        pass


@pytest.fixture()
def only_fit_dummy_anonymizer():
    return OnlyFitDummy()


@pytest.fixture()
def only_predict_dummy_anonymizer():
    return OnlyPredictDummy()


@pytest.fixture()
def partial_dummy_anonymizer():
    return PartialDummy()


@pytest.fixture()
def complete_dummy_anonymizer():
    return CompleteDummy()


def dummy_validation_function(_: np.ndarray) -> None:
    pass


def test_WrappedAnonymizer(
    partial_dummy_anonymizer: PartialDummy,
    complete_dummy_anonymizer: CompleteDummy,
    empty_ndarray: np.ndarray,
    ten_elem_ndarray: np.ndarray,
    ten_ones_ndarray: np.ndarray,
    tmp_path: Path,
):
    for wrapped_anon in [
        anonymizer.WrappedAnonymizer(complete_dummy_anonymizer),
        anonymizer.WrappedAnonymizer(
            partial_dummy_anonymizer,
            serializer=anonymizer.pickle_anonymizer,
            feature_array_validator=dummy_validation_function,
        ),
    ]:
        wrapped_anon.fit(ten_elem_ndarray, ten_elem_ndarray)
        assert np.array_equal(wrapped_anon.predict(empty_ndarray), empty_ndarray)
        assert np.array_equal(wrapped_anon.predict(ten_elem_ndarray), ten_ones_ndarray)
        assert np.array_equal(
            wrapped_anon.predict(ten_elem_ndarray, 4), ten_ones_ndarray
        )
        p = tmp_path / "foo.bar"
        wrapped_anon.save(p)
        assert p.exists()
        unpickled_anon = anonymizer.unpickle_anonymizer(p)
        assert hasattr(unpickled_anon, "fit")
        assert hasattr(unpickled_anon, "predict")
        wrapped_anon.validate_input(empty_ndarray)


def test_WrappedAnonymizer_sklearn(ten_elem_ndarray, tmp_path: Path):
    X = np.arange(100, dtype=np.float_).reshape(10, 10)
    gnb = anonymizer.WrappedAnonymizer(
        GaussianNB(),
        serializer=anonymizer.pickle_anonymizer,
        feature_array_validator=dummy_validation_function,
    )
    p = tmp_path / "model.pkl"
    gnb.save(p)
    anons = [gnb, anonymizer.unpickle_anonymizer(p)]
    for anon in anons:
        anon.fit(X=X, y=ten_elem_ndarray)
        anon.predict(X=X, batch_size=4)


def test_raising_WrappedAnonymizer(
    only_fit_dummy_anonymizer: OnlyFitDummy,
    only_predict_dummy_anonymizer: OnlyPredictDummy,
    partial_dummy_anonymizer: PartialDummy,
    empty_ndarray,
):
    with pytest.raises(NotImplementedError):
        anonymizer.WrappedAnonymizer(only_fit_dummy_anonymizer)
    with pytest.raises(NotImplementedError):
        anonymizer.WrappedAnonymizer(only_predict_dummy_anonymizer)
    with pytest.raises(NotImplementedError):
        anonymizer.WrappedAnonymizer(partial_dummy_anonymizer).save("")
    with pytest.raises(NotImplementedError):
        anonymizer.WrappedAnonymizer(partial_dummy_anonymizer).validate_input(
            empty_ndarray
        )


def test_TFKerasWrapper(ten_elem_ndarray, tmp_path):
    tfkeras = pytest.importorskip("tf_keras")
    model = tfkeras.Sequential(
        [
            tfkeras.layers.Input(shape=(1,)),
            tfkeras.layers.Dense(10, activation="relu"),
            tfkeras.layers.Dense(1, activation="relu"),
        ]
    )
    model(ten_elem_ndarray)
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[tfkeras.metrics.MeanAbsoluteError(name="mean_absolute_error")],
    )

    additional_fit_args = dict(epochs=3)

    wrapped_model = anonymizer.TFKerasWrapper(
        tfkeras_model=model,
        feature_array_validator=dummy_validation_function,
        **additional_fit_args,
    )
    p = tmp_path / "model.keras"
    wrapped_model.save(p)
    saved_model = anonymizer.TFKerasWrapper(
        tfkeras_model=tfkeras.models.load_model(p),
        feature_array_validator=dummy_validation_function,
        **additional_fit_args,
    )
    for anon in [wrapped_model, saved_model]:
        anon.fit(X=ten_elem_ndarray, y=ten_elem_ndarray)
        anon.predict(ten_elem_ndarray)
        anon.validate_input(ten_elem_ndarray)


def test_TorchAnonymizer(ten_elem_ndarray, tmp_path):
    torch = pytest.importorskip("torch")

    class TorchModel(anonymizer.SupervisedLearningAnonymizer):
        def __init__(self) -> None:
            self._model = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> None:
            X_ = torch.tensor(X, dtype=torch.float32).reshape(-1, 1)
            y_ = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
            model = torch.nn.Sequential(
                torch.nn.Linear(in_features=1, out_features=10, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=10, out_features=1, bias=True),
                torch.nn.ReLU(),
            )
            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            y_pred = model(X_)
            loss = loss_fn(y_pred, y_)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._model = model

        def predict(self, X: np.ndarray, batch_size: int | None = None) -> np.ndarray:
            if self._model is None:
                raise NotImplementedError(
                    "Model not fitted yet. No prediction possible."
                )
            self._model.eval()
            predictions = []
            with torch.no_grad():
                for X_batch in anonymizer.batch_views(X, batch_size):
                    batch_tensor = torch.tensor(X_batch, dtype=torch.float).reshape(
                        -1, 1
                    )
                    prediction = self._model(batch_tensor)
                    predictions.append(prediction.numpy())
            return np.concatenate(predictions)

        def save(self, filepath: str | Path) -> None:
            torch.save(self._model, filepath)

        def validate_input(self, feature_array: np.ndarray) -> None:
            pass

    def load_torch_model(filepath: str | Path) -> TorchModel:
        model = TorchModel()
        model._model = torch.load(filepath, weights_only=False)
        return model

    wrapped_model = TorchModel()
    p = tmp_path / "model.pt"
    wrapped_model.save(p)
    saved_model = load_torch_model(p)
    for anon in [wrapped_model, saved_model]:
        anon.fit(X=ten_elem_ndarray, y=ten_elem_ndarray)
        anon.predict(ten_elem_ndarray)
        anon.validate_input(ten_elem_ndarray)


def test_batch_views(empty_ndarray: np.ndarray, ten_elem_ndarray: np.ndarray):
    [batch] = anonymizer.batch_views(empty_ndarray, 0)
    assert len(batch) == 0

    assert [] == anonymizer.batch_views(ten_elem_ndarray, -1)
    assert [] == anonymizer.batch_views(ten_elem_ndarray, 0)

    [batch] = anonymizer.batch_views(ten_elem_ndarray, None)
    assert np.array_equal(batch, ten_elem_ndarray)

    [batch] = anonymizer.batch_views(ten_elem_ndarray, 10)
    assert np.array_equal(batch, ten_elem_ndarray)

    [batch] = anonymizer.batch_views(ten_elem_ndarray, 11)
    assert np.array_equal(batch, ten_elem_ndarray)

    batches = anonymizer.batch_views(ten_elem_ndarray, 3)
    assert np.array_equal(np.concatenate(batches), ten_elem_ndarray)
