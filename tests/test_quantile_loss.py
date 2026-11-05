import numpy as np
from spev.metrics import _quantile_loss

def test_equals_twice_the_quantile_level_when_larger_by_one():
    result = _quantile_loss(
        y_true = np.zeros((1, 1, 1, 1)),
        y_pred = np.ones((1, 2501, 1, 1)),
        quantile_level = 0.9,
    )[0, 0, 0]

    assert result == 1.8

    result = _quantile_loss(
        y_true = np.zeros((1, 1, 1, 1)),
        y_pred = np.ones((1, 2501, 1, 1)),
        quantile_level = 0.1,
    )[0, 0, 0]

    assert result == 0.2

def test_equals_twice_one_minus_quantile_level_when_less_by_one():
    result = _quantile_loss(
        y_true = np.ones((1, 1, 1, 1)),
        y_pred = np.zeros((1, 2501, 1, 1)),
        quantile_level = 0.9,
    )[0, 0, 0]

    assert result == ((1.0 - 0.9) * 2.0)

    result = _quantile_loss(
        y_true = np.ones((1, 1, 1, 1)),
        y_pred = np.zeros((1, 2501, 1, 1)),
        quantile_level = 0.1,
    )[0, 0, 0]

    assert result == ((1.0 - 0.1) * 2.0)

def test_equals_one_when_median_and_off_by_one():
        result = _quantile_loss(
            y_true = np.ones((1, 1, 1, 1)),
            y_pred = np.zeros((1, 2501, 1, 1)),
            quantile_level = 0.5,
        )[0, 0, 0]

        assert result == 1.0

        result = _quantile_loss(
            y_true = np.zeros((1, 1, 1, 1)),
            y_pred = np.ones((1, 2501, 1, 1)),
            quantile_level = 0.5,
        )[0, 0, 0]

        assert result == 1.0