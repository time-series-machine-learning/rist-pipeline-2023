# Extracting Features from Random Subseries: A Hybrid Pipeline for Time Series Classification and Extrinsic Regression

Code and results for the paper "Extracting Features from Random Subseries: A Hybrid
Pipeline for Time Series Classification and Extrinsic Regression" from the
International Workshop on Advanced Analytics and Learning on Temporal Data (AALTD)
2023, which introduces the RIST pipeline.

This material was previously part of
[tsml-eval](https://github.com/time-series-machine-learning/tsml-eval) under
`tsml_eval/publications/y2023/rist_pipeline`.

## Contents

| Path | Description |
|---|---|
| [results/classification](results/classification) | Accuracy for each classifier on the 112 UCR datasets, one column per resample |
| [results/regression](results/regression) | RMSE for each regressor on the 63 regression datasets, one column per resample |
| [set_rist_classifier.py](set_rist_classifier.py), [set_rist_regressor.py](set_rist_regressor.py) | The estimator configurations used in the experiments |
| [run_classification_experiments.py](run_classification_experiments.py), [run_regression_experiments.py](run_regression_experiments.py) | Scripts to run a single experiment |
| [rist_pipeline.ipynb](rist_pipeline.ipynb) | Example of using the RIST classifier and regressor directly |
| [static_publication_reqs.txt](static_publication_reqs.txt) | Dependency versions used at the time of publication |

## Datasets

The 112 UCR archive datasets are available at
[timeseriesclassification.com](http://www.timeseriesclassification.com/dataset.php).

The 63 regression datasets are available from the
[TSER archive expansion webpage](https://tsml-eval.readthedocs.io/en/latest/publications/2023/tser_archive_expansion/tser_archive_expansion.html).

## Running the experiments

There are two ways to run the code. The published results were generated with the
versions described in [As published](#as-published). Results from current versions
may differ, as the algorithms have changed since publication.

### Latest versions

Clone this repository and install the latest `tsml-eval` with all optional estimator
dependencies:

    pip install -r requirements.txt

For a smaller install, use `pip install tsml-eval tsml` and add the dependencies for
individual estimators as required. RIST requires `statsmodels`.

Run [run_classification_experiments.py](run_classification_experiments.py) or
[run_regression_experiments.py](run_regression_experiments.py) from the repository
root with the following arguments:

1. Path to the data directory
2. Path to the results directory
3. The name of the model to run (see [set_rist_classifier.py](set_rist_classifier.py)
   or [set_rist_regressor.py](set_rist_regressor.py), i.e. RIST, RDST, DrCIF)
4. The name of the problem to run
5. The resample number to run (0 is base train/test split)

i.e. to run the ItalyPowerDemand classification problem using RIST on the base
train/test split:

    python run_classification_experiments.py data/ results/ RIST ItalyPowerDemand 0

Running either script without arguments runs ROCKET on a small example dataset from
[tests/data](tests/data) and writes to `test_output/`.

The code in this repository was last checked in November 2026 with `tsml-eval` 0.7.0,
`aeon` 1.5.0 and `tsml` 0.10.0 on Python 3.12. It is not actively maintained against
new releases.

### As published

The experiments for the paper were run from the `tsml-eval` source using `aeon` 0.3.0
and `tsml` 0.2.1. The code in this repository has since been updated for newer
versions, so to run the original code use the `tsml-eval` repository at the last
commit which used these versions. This requires Python 3.8 to 3.11.

    git clone https://github.com/time-series-machine-learning/tsml-eval.git
    cd tsml-eval
    git checkout 1efd1b46
    pip install -r tsml_eval/publications/y2023/rist_pipeline/requirements.txt
    pip install --no-deps .
    python tsml_eval/publications/y2023/rist_pipeline/run_classification_experiments.py data/ results/ RIST ItalyPowerDemand 0

The requirements file is the same as
[static_publication_reqs.txt](static_publication_reqs.txt) in this repository. It
was generated on Windows, so `pywin32`, `tensorflow-intel`, `ansicon` and `jinxed`
should be removed on other operating systems.

At the time of publication, RIST required the `pycatch22` and `pyfftw` packages. To
install these, run:

    pip install pycatch22 pyfftw

This can be unstable on some setups. If you cannot install these packages, they can
be disabled in the publication version by editing the classifier parameters (this
will change the results produced) i.e.

    RISTClassifier(use_pycatch22=False, use_pyfftw=False)

Current versions of `aeon` do not use these packages for RIST by default, and the
`use_pyfftw` parameter has been removed.

## Using the classifiers and regressors

The estimators used in the experiments are available in the
[aeon](https://github.com/aeon-toolkit/aeon) and
[tsml](https://github.com/time-series-machine-learning/tsml-py) Python packages and
follow the `scikit-learn` interface. See [rist_pipeline.ipynb](rist_pipeline.ipynb)
for an example.
