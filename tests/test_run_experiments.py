"""Test file checking that the RIST pipeline experiments run correctly."""

import os
import runpy
import sys

from tsml_eval.testing.testing_utils import _TEST_DATA_PATH
from tsml_eval.utils.results_validation import validate_results_file

from run_classification_experiments import (
    _REPO_PATH,
    _RIST_TEST_RESULTS_PATH,
    _run_classification_experiment,
)
from run_regression_experiments import _run_regression_experiment


def test_run_rist_pipeline_classification_experiment(monkeypatch):
    """Test paper classification experiments with test data and classifier."""
    classifier = "ROCKET"
    dataset = "MinimalChinatown"
    resample = 0

    args = [
        _TEST_DATA_PATH,
        _RIST_TEST_RESULTS_PATH,
        classifier,
        dataset,
        resample,
        "-ow",
    ]

    _run_classification_experiment(args)

    test_file = (
        f"{_RIST_TEST_RESULTS_PATH}{classifier}/Predictions/{dataset}/testResample0.csv"
    )
    assert os.path.exists(test_file)
    assert validate_results_file(test_file)

    # this covers both the main method and present result file checking
    monkeypatch.setattr(sys, "argv", sys.argv[:1])
    runpy.run_path(
        f"{_REPO_PATH}/run_classification_experiments.py", run_name="__main__"
    )

    os.remove(test_file)


def test_run_rist_pipeline_regression_experiment(monkeypatch):
    """Test paper regression experiments with test data and regressor."""
    regressor = "ROCKET"
    dataset = "MinimalGasPrices"
    resample = 0

    args = [
        _TEST_DATA_PATH,
        _RIST_TEST_RESULTS_PATH,
        regressor,
        dataset,
        resample,
        "-ow",
    ]

    _run_regression_experiment(args)

    test_file = (
        f"{_RIST_TEST_RESULTS_PATH}{regressor}/Predictions/{dataset}/testResample0.csv"
    )
    assert os.path.exists(test_file)
    assert validate_results_file(test_file)

    # this covers both the main method and present result file checking
    monkeypatch.setattr(sys, "argv", sys.argv[:1])
    runpy.run_path(f"{_REPO_PATH}/run_regression_experiments.py", run_name="__main__")

    os.remove(test_file)
