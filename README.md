[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![pipeline status](https://git.uni-luebeck.de/its/anomed/anonymizer/badges/main/pipeline.svg?ignore_skipped=true)
![coverage](https://git.uni-luebeck.de/its/anomed/anonymizer/badges/main/coverage.svg?job=run_tests)

# Anonymizer

A library aiding to create anonymizers (privacy preserving machine learning
models) for the AnoMed competition platform.

# Usage Example

The following example will create a Falcon-based web app that encapsulates an
anonymizer for an example challenge (which serves the famous iris dataset). The
encapsulated anonymizer is differentially private Gaussian naive Bayes
classifier, which aims to solve a 3-class classification problem.

The web app offers these routes (some may have query parameters not mentioned
here):

- [GET] `/` (This displays an "alive message".)
- [POST] `/fit` (This invokes fitting the Gaussian naive based classifier;
  the web app will pull the training data from `training_data_url`.)
- [POST] `/evaluate` (This invokes an intermediate, or final evaluation of the
  classifier.)
- [POST] `/predict` (This offers a way to use the fitted anonymizer to predict
  the target values for arbitrary, but compatible, feature arrays.)

```python
import anomed_anonymizer as anon
import numpy as np
from diffprivlib.models import GaussianNB

lower_bounds = 4 * [0.0]
upper_bounds = [10.0, 5.0, 10.0, 5.0]
estimator = GaussianNB(
    bounds=(lower_bounds, upper_bounds),
    priors=3 * [1.0 / 3.0],
)


def input_array_validator(feature_array: np.ndarray) -> None:
    if feature_array.shape[1] != 4 or len(feature_array.shape) != 2:
        raise ValueError("Feature array needs to have shape (n_samples, 4).")
    if feature_array.dtype != np.float_:
        raise ValueError("Feature array must be an array of floats.")


example_anon = anon.WrappedAnonymizer(
    anonymizer=estimator,
    serializer=anon.pickle_anonymizer,
    feature_array_validator=input_array_validator,
)

hostname = "example.com"

# This is what GUnicorn expects
application = anon.supervised_learning_anonymizer_server_factory(
    anonymizer_identifier="example_anonymizer",
    anonymizer_obj=example_anon,
    model_filepath="anonymizer.pkl",
    default_batch_size=64,
    training_data_url=f"http://{hostname}/data/anonymizer/training",
    tuning_data_url=f"http://{hostname}/data/anonymizer/tuning",
    validation_data_url=f"http://{hostname}/data/anonymizer/training",
    utility_evaluation_url=f"http://{hostname}/utility/anonymizer",
    model_loader=anon.unpickle_anonymizer,
)
```
