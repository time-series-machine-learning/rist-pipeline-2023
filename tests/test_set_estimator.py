"""Tests for publication experiments estimator selection."""

import pytest
from sklearn.base import BaseEstimator

from set_rist_classifier import _set_rist_classifier, rist_classifiers
from set_rist_regressor import _set_rist_regressor, rist_regressors


def _names(estimator_names):
    names = []
    for n in estimator_names:
        names.extend([n] if isinstance(n, str) else n)
    return names


@pytest.mark.parametrize("name", _names(rist_classifiers))
def test_set_rist_classifier(name):
    """Test set_rist_classifier method."""
    try:
        classifier = _set_rist_classifier(name)
    except ModuleNotFoundError as err:
        pytest.skip(f"Optional dependency not installed: {err}")
    assert isinstance(classifier, BaseEstimator)


def test_set_rist_classifier_invalid():
    """Test set_rist_classifier method with invalid estimator."""
    with pytest.raises(ValueError, match="UNKNOWN CLASSIFIER"):
        _set_rist_classifier("invalid")


@pytest.mark.parametrize("name", _names(rist_regressors))
def test_set_rist_regressor(name):
    """Test set_rist_regressor method."""
    try:
        regressor = _set_rist_regressor(name)
    except ModuleNotFoundError as err:
        pytest.skip(f"Optional dependency not installed: {err}")
    assert isinstance(regressor, BaseEstimator)


def test_set_rist_regressor_invalid():
    """Test set_rist_regressor method with invalid estimator."""
    with pytest.raises(ValueError, match="UNKNOWN REGRESSOR"):
        _set_rist_regressor("invalid")
