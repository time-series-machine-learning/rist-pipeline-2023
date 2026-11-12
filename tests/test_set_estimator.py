"""Tests for publication experiments estimator selection."""

import numpy as np
import pytest
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from tsml_eval.testing.testing_utils import _check_set_method, _check_set_method_results

from set_rist_classifier import _set_rist_classifier, rist_classifiers
from set_rist_regressor import _set_rist_regressor, rist_regressors


def test_set_rist_classifier():
    """Test set_rist_classifier method."""
    classifier_list = []
    classifier_dict = {}
    all_classifier_names = []
    _check_set_method(
        _set_rist_classifier,
        rist_classifiers,
        classifier_list,
        classifier_dict,
        all_classifier_names,
    )

    _check_set_method_results(
        classifier_dict,
        estimator_name="Classifiers",
        method_name="_set_rist_classifier",
    )


def test_set_rist_classifier_invalid():
    """Test set_rist_classifier method with invalid estimator."""
    with pytest.raises(ValueError, match="UNKNOWN CLASSIFIER"):
        _set_rist_classifier("invalid")


def test_set_rist_regressor():
    """Test set_rist_regressors method."""
    regressor_list = []
    regressor_dict = {}
    all_regressor_names = []
    _check_set_method(
        _set_rist_regressor,
        rist_regressors,
        regressor_list,
        regressor_dict,
        all_regressor_names,
    )

    _check_set_method_results(
        regressor_dict, estimator_name="Regressors", method_name="_set_rist_regressor"
    )


def test_set_rist_regressor_invalid():
    """Test set_rist_regressor method with invalid estimator."""
    with pytest.raises(ValueError, match="UNKNOWN REGRESSOR"):
        _set_rist_regressor("invalid")


@pytest.mark.parametrize(
    "name,fast_params",
    [
        (
            "RIST-ExtraT",
            {
                "n_intervals": 2,
                "n_shapelets": 5,
                "estimator": ExtraTreesClassifier(n_estimators=5, criterion="entropy"),
            },
        ),
        (
            "RIST-RF",
            {
                "n_intervals": 2,
                "n_shapelets": 5,
                "estimator": RandomForestClassifier(
                    n_estimators=5, criterion="entropy"
                ),
            },
        ),
        ("RIST-RidgeCV", {"n_intervals": 2, "n_shapelets": 5}),
        ("RDST", {"max_shapelets": 10}),
        ("R-STSF", {"n_estimators": 5, "n_intervals": 2}),
        ("DrCIF", {"n_estimators": 3, "n_intervals": 2, "att_subsample_size": 2}),
        ("ROCKET", {"n_kernels": 20}),
        ("FreshPRINCE", {"n_estimators": 5, "default_fc_parameters": "minimal"}),
    ],
)
def test_fit_rist_classifiers(name, fast_params):
    """Test fitting and predicting with RIST publication classifiers on short series."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 2, 24))
    y = np.array([0, 1] * 6)

    try:
        classifier = _set_rist_classifier(name, random_state=0)
    except ModuleNotFoundError as err:
        pytest.skip(f"Optional dependency not installed: {err}")

    classifier.set_params(**fast_params)
    classifier.fit(X, y)
    preds = classifier.predict(X)
    assert preds.shape == (12,)


@pytest.mark.parametrize(
    "name,fast_params",
    [
        (
            "RIST-ExtraT",
            {
                "n_intervals": 2,
                "n_shapelets": 5,
                "estimator": ExtraTreesRegressor(n_estimators=5),
            },
        ),
        (
            "RIST-RF",
            {
                "n_intervals": 2,
                "n_shapelets": 5,
                "estimator": RandomForestRegressor(n_estimators=5),
            },
        ),
        ("RIST-RidgeCV", {"n_intervals": 2, "n_shapelets": 5}),
        ("RDST", {"max_shapelets": 10}),
        ("DrCIF", {"n_estimators": 3, "n_intervals": 2, "att_subsample_size": 2}),
        ("ROCKET", {"n_kernels": 20}),
        ("FreshPRINCE", {"n_estimators": 5, "default_fc_parameters": "minimal"}),
    ],
)
def test_fit_rist_regressors(name, fast_params):
    """Test fitting and predicting with RIST publication regressors."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 2, 24))
    y = rng.normal(size=12)

    try:
        regressor = _set_rist_regressor(name, random_state=0)
    except ModuleNotFoundError as err:
        pytest.skip(f"Optional dependency not installed: {err}")

    regressor.set_params(**fast_params)
    regressor.fit(X, y)
    preds = regressor.predict(X)
    assert preds.shape == (12,)
