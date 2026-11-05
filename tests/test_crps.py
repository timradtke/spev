import numpy as np
from spev.metrics import _crps

def crps_naive(y_true, y_pred):
    """Memory-inefficient but simpler variant of CRPS."""
    mae = np.mean(np.abs(y_true - y_pred))
    pairwise_differences = np.mean(np.abs(y_pred[:, None] - y_pred[None, :]))
    return mae - 0.5 * pairwise_differences

def test_reproduces_reference_implementation():
    y_true = np.array([10])
    y_pred = np.array([2, 5, 11, 13])

    expected = crps_naive(y_true, y_pred)
    result = _crps(
        y_true = y_true[None, :, None, None],
        y_pred = y_pred[None, :, None, None],
    )[0, 0, 0]

    assert np.abs(result - expected) < 0.00000000001

def test_reproduces_reference_implementation_when_negative_actuals():
    y_true = np.array([-5.5])
    y_pred = np.array([2, 5, 11, 13])

    expected = crps_naive(y_true, y_pred)
    result = _crps(
        y_true = y_true[None, :, None, None],
        y_pred = y_pred[None, :, None, None],
    )[0, 0, 0]

    assert np.abs(result - expected) < 0.00000000001

def test_reproduces_reference_implementation_mixed_signs():
    y_true = np.array([4.9])
    y_pred = np.array([-20.0, -1.5, 2, 4.9, 5, 11, 13])

    expected = crps_naive(y_true, y_pred)
    result = _crps(
        y_true = y_true[None, :, None, None],
        y_pred = y_pred[None, :, None, None],
    )[0, 0, 0]

    assert np.abs(result - expected) < 0.00000000001

def test_reproduces_reference_implementation_mixed_signs_and_shuffled():
    y_true = np.array([4.9])
    y_pred = np.array([13, -20.0, 11, -1.5, 4.9, 2, 5])

    expected = crps_naive(y_true, y_pred)
    result = _crps(
        y_true = y_true[None, :, None, None],
        y_pred = y_pred[None, :, None, None],
    )[0, 0, 0]

    assert np.abs(result - expected) < 0.00000000001

def equals_mae_when_samples_are_constant():
    y_true = np.array([5.0])
    y_pred = np.array([7.5, 7.5, 7.5, 7.5, 7.5])

    result = _crps(
        y_true = y_true[None, :, None, None],
        y_pred = y_pred[None, :, None, None],
    )[0, 0, 0]

    assert result == 2.5

    y_true = np.array([-5.0])
    y_pred = np.array([7.5, 7.5, 7.5, 7.5, 7.5])

    result = _crps(
        y_true = y_true[None, :, None, None],
        y_pred = y_pred[None, :, None, None],
    )[0, 0, 0]

    assert result == 12.5
