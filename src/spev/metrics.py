from typing import Any, Type
import numpy as np

class SamplePathMetric:
    """Base class for all metrics."""
    @property
    def name(self) -> str:
        """Name of the metric."""
        return self.__class__.__name__

    @staticmethod
    def _safemean(arr: np.ndarray, axis=None) -> float | np.ndarray:
        """Compute mean ignoring NaN, Inf, and -Inf values."""
        mask = ~np.isfinite(arr)
        if mask.any():
            arr = np.where(mask, np.nan, arr)
        return np.nanmean(arr, axis=axis)

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        reduce_axes: tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        raise NotImplementedError

class MeanCRPS(SamplePathMetric):
    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        reduce_axes: tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        crps = _crps( # (N, H, D)
            y_true=y_true,
            y_pred=y_pred,
        )
        return np.nanmean(crps, axis=reduce_axes)  # [..., D]

class WeightedCRPS(SamplePathMetric):
    def __init__(self, epsilon: float = 0.0) -> None:
            self.epsilon = epsilon

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        reduce_axes: tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        crps = _crps( # (N, H, D)
            y_true=y_true,
            y_pred=y_pred,
        )
        crps_agg = np.nanmean(crps, axis=reduce_axes)  # [..., D]
        abs_true = np.nanmean(np.abs(y_true[:, 0, :, :]), axis=reduce_axes)  # [..., D]
        return crps_agg / np.maximum(abs_true, self.epsilon)  # [..., D]

class MeanQuantileLoss(SamplePathMetric):
    def __init__(self, quantile_level: float) -> None:
        self.quantile_level = quantile_level

    @property
    def name(self) -> str:
        """Name of the metric."""
        return self.__class__.__name__ + "@" + str(self.quantile_level)

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        reduce_axes: tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        ql = _quantile_loss( # (N, H, D)
            y_true=y_true,
            y_pred=y_pred,
            quantile_level=self.quantile_level
        )
        return np.nanmean(ql, axis=reduce_axes)  # [..., D]

class WeightedQuantileLoss(SamplePathMetric):
    def __init__(self, quantile_level: float, epsilon: float = 0.0) -> None:
        self.quantile_level = quantile_level
        self.epsilon = epsilon

    @property
    def name(self) -> str:
        """Name of the metric."""
        return self.__class__.__name__ + "@" + str(self.quantile_level)

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        reduce_axes: tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        ql = _quantile_loss( # (N, H, D)
            y_true=y_true,
            y_pred=y_pred,
            quantile_level=self.quantile_level
        )
        ql_agg = np.nanmean(ql, axis=reduce_axes)  # [..., D]
        abs_true = np.nanmean(np.abs(y_true[:, 0, :, :]), axis=reduce_axes)  # [..., D]
        return ql_agg / np.maximum(abs_true, self.epsilon)  # [..., D]

def _quantile_loss(
    y_true: np.ndarray, # (N, S, H, D)
    y_pred: np.ndarray, # (N, S, H, D)
    quantile_level: float,
) -> np.ndarray: # (N, H, D)

    q_true = np.quantile(
        a=y_true,
        q=quantile_level,
        axis=1,
    )

    q_pred = np.quantile(
        a=y_pred,
        q=quantile_level,
        axis=1,
    )
    
    return (
        2 * (q_pred - q_true) * (
            quantile_level * (q_pred > q_true) - \
                (1. - quantile_level) * (q_pred <= q_true)
        )
    )

def _crps(
    y_true, # (N, S, H, D)
    y_pred, # (N, S, H, D)
) -> np.ndarray: # (N, H, D)
    # See:
    # https://skforecast.org/0.15.0/faq/probabilistic-forecasting-crps-score.html
    # https://www.lokad.com/continuous-ranked-probability-score/
    # https://getrecast.com/crps-evaluate-recast-models/
    S = y_pred.shape[1]
    y_pred_sorted = np.sort(y_pred, axis=1) # Sort the samples for a given (N, H, D)
        
    mae = np.mean(np.abs(y_pred - y_true), axis=1) # (N, H, D)

    c = np.arange(1, S+1)
    weights = 2 * c - S - 1 # (S)
    weights = weights[None, ..., None, None] # (N, S, H, D)
    pairwise_diff = np.sum(weights * y_pred_sorted, axis=1) / (S ** 2) # (N, H, D)

    crps = mae - pairwise_diff
    return crps

MetricConfig = str | dict[str, Any]

def get_metric(metric: MetricConfig) -> SamplePathMetric:
    """Get a metric class by name or configuration."""
    metric_name = metric if isinstance(metric, str) else metric["name"]
    try:
        metric_type = AVAILABLE_METRICS[metric_name.upper()]
    except KeyError:
        raise ValueError(
            f"Evaluation metric '{metric_name}' is not available. Available metrics: {sorted(AVAILABLE_METRICS)}"
        )

    if isinstance(metric, str):
        return metric_type()
    elif isinstance(metric, dict):
        return metric_type(**{k: v for k, v in metric.items() if k != "name"})
    else:
        raise ValueError(f"Invalid metric configuration: {metric}")

AVAILABLE_METRICS: dict[str, Type[SamplePathMetric]] = {
    "MCRPS": MeanCRPS,
    "WCRPS": WeightedCRPS,
    "MQL": MeanQuantileLoss,
    "WQL": WeightedQuantileLoss,
}
