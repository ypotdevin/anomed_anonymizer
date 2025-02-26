# Anonymizer

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A library aiding to create anonymizers (privacy preserving machine learning
models) for the AnoMed competition platform.

## Preliminaries

The AnoMed platform is basically a network of web servers which use web APIs to
exchange data among each other and provide functionality to each other.
[Challenge web servers](https://github.com/ypotdevin/anomed_challenge) provide
training and evaluation data, which may be requested via HTTP. They do also
offer means to evaluate the utility of anonymizers (privacy preserving machine
learning models) via HTTP and means to estimate the privacy of anonymizers via
[attacks](https://github.com/ypotdevin/anomed_deanonymizer) on them (which we
refer to by "deanonymizers" below). Anonymizer web servers offer input/output
access, such that they may be attacked by deanonymizers. For more details about
challenges or deanonymizers, view their corresponding repositories.

In general, you are free to create your own kind of anonymizer web server, as
long as it offers some well described APIs and follows some general principles,
which we will describe below. You do not need to use this library to submit
anonymizers. However, if you would like to focus on defining the anonymizer
itself, without being annoyed by web server related questions, use this library
to generate web servers "for free", which integrate well with the AnoMed
platform. If you plan to contribute to a challenge that uses one of [our
challenge web server templates](https://github.com/ypotdevin/anomed_challenge),
we very strongly suggest to make use of the accompanying anonymizer web server
template provided in this repository, as all web-related issues have been taken
care of by us already.

## How to Create Anonymizer Web Servers (for selected use cases)

If you goal is to create an anonymizer that fits one of the following selected
cases, you may use this library's template to create an anonymizer web server
with minimal effort.

In the following we give some examples of how to use this library to create
anonymizer submissions for the AnoMed platform. First, we cover some common
cases for which we have created templates and then we tell you what to do, if
these template do not suite your use case.

### Anonymizer for Supervised Learning Challenges with Membership Inference Attack Threat Model

In this scenario we assume that the challenge you would like to contribute to
has been created by the
[`supervised_learning_MIA_challenge_server_factory`](https://anomed-challenge.readthedocs.io/en/latest/apidocs/anomed_challenge/anomed_challenge.challenge_server.html#anomed_challenge.challenge_server.supervised_learning_MIA_challenge_server_factory)
function. Also, we assume that the challenge serves the [Iris
dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris),
just like in the [usage
example](https://github.com/ypotdevin/anomed_challenge?tab=readme-ov-file#supervised-learning-challenge-with-membership-inference-attack-threat-model)
of the AnoMed Challenge library. Now we know which routes are present, how to
interact with them and what data to expect. The function
[`supervised_learning_anonymizer_server_factory`](https://anomed-anonymizer.readthedocs.io/en/latest/apidocs/anomed_anonymizer/anomed_anonymizer.anonymizer_server.html#anomed_anonymizer.anonymizer_server.supervised_learning_anonymizer_server_factory)
will create a suitable [Falcon-based](https://falcon.readthedocs.io/en/stable/)
anonymizer web server for us. In the following example, we show how to use it:

```python
import os

import anomed_anonymizer as anon
import numpy as np
from diffprivlib.models import GaussianNB

lower_bounds = 4 * [0.0]
upper_bounds = [10.0, 5.0, 10.0, 5.0]
estimator = GaussianNB(
    bounds=(lower_bounds, upper_bounds),
    priors=3 * [1.0 / 3.0],
)


def validate_feature_array(feature_array: np.ndarray) -> None:
    if feature_array.shape[1] != 4 or len(feature_array.shape) != 2:
        raise ValueError("Feature array needs to have shape (n_samples, 4).")
    if feature_array.dtype != np.float_:
        raise ValueError("Feature array must be an array of floats.")


example_anon = anon.WrappedAnonymizer(
    anonymizer=estimator,
    serializer=anon.pickle_anonymizer,
    feature_array_validator=validate_feature_array,
)

hostname = os.getenv("CHALLENGE_HOST")

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

First, we create an allegedly privacy-preserving machine learning model – in
this case, for example, an instance of
[`GaussianNB`](https://diffprivlib.readthedocs.io/en/latest/modules/models.html#gaussian-naive-bayes).
That model is a differentially private Gaussian naive Bayes classifier, which
aims to solve a 3-class classification problem. Next, we define a feature array
validation function that double-checks, whether the data received from the
challenge is of the expected shape and type. See the docs of
[`WrappedAnonymizer`](https://anomed-anonymizer.readthedocs.io/en/latest/apidocs/anomed_anonymizer/anomed_anonymizer.anonymizer.html#anomed_anonymizer.anonymizer.WrappedAnonymizer)
for more details. After that, we wrap the model, so that we receive an object of
type
[`SupervisedLearningAnonymizer`](https://anomed-anonymizer.readthedocs.io/en/latest/apidocs/anomed_anonymizer/anomed_anonymizer.anonymizer.html#anomed_anonymizer.anonymizer.SupervisedLearningAnonymizer),
which is a necessary argument of
`supervised_learning_anonymizer_server_factory`. Then we obtain the hostname of
the challenge we would like to contribute to from an environment variable, which
is set to the correct value by the AnoMed platform. Finally, we create the web
application using `supervised_learning_anonymizer_server_factory` (for the
meaning of the other arguments, [see
here](https://anomed-anonymizer.readthedocs.io/en/latest/apidocs/anomed_anonymizer/anomed_anonymizer.anonymizer_server.html#anomed_anonymizer.anonymizer_server.supervised_learning_anonymizer_server_factory)).
Use `application` as a target for, e.g., GUnicorn to serve the anonymizer web
application.

The web app offers these routes (some may have query parameters not mentioned
here):

- [GET] `/`: This displays an "alive message".
- [POST] `/fit`: This invokes fitting the Gaussian naive based classifier;
  the web app will then pull the training data from `training_data_url`.
- [POST] `/evaluate`: This invokes an intermediate, or final evaluation of the
  classifier, depending on query parameters.
- [POST] `/predict`: This offers a way to use the fitted anonymizer to predict
  the target values for any compatible feature array.

### Anonymizer for Dataset Anonymization Challenges with ??? Threat Model

TODO

### Anonymizer for Dataset Synthesis Challenges with ??? Threat Model

TODO

## How To Create Challenge Web Servers Without Template

In case your goal is to contribute to a challenge, for which we do not offer a
suitable anonymizer template, we suggest that you stick to the Falcon web
framework and make use of at least some of the available resource building
blocks. Besides that, you should pay attention to the following principles when
implementing your anonymizer:

- Challenges and submissions will not get any internet access when running on
  the AnoMed platform. Make your anonymizer self-containing.
- Explain your API well in the anonymizer description, such that custom
  deanonymizers have it easy to obey your API. [Template
  deanonymizers](https://github.com/ypotdevin/anomed_deanonymizer) are likely
  incompatible with your custom anonymizer.
- Provide a default route `GET /` which returns a JSON encoded message like
  "Anonymizer server is alive!" for diagnosis, upon request.
- Provide a route `POST /fit` which starts the the fitting process upon request.
  If fitting has been successful, respond with a `201 Created`. If `POST /fit`
  is invoked while a fitting is already in progress, respond with
  `503 Service Unavailable`.
- Provide a route `POST /evaluate` which triggers the evaluation of your
  anonymizer. Depending on the specific challenge, you might have to expect
  query parameters like `data_split` which differentiate the kind of evaluation
  (e.g. intermediate vs final evaluation).
