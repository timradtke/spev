from .metrics import MetricConfig, get_metric
from fev import Task as FevTask
from typing import Any
import numpy as np
import dataclasses

@dataclasses.dataclass
class Task:
    """
    """

    def __init__(
        self,
        base: FevTask,
        metrics: list[dict[MetricConfig]],
        #[
        #    {
        #        "name": "WCRPS",
        #        "epsilon": 0.01,
        #    },
        #    {
        #        "name": "WQL",
        #        "quantile_level": 0.9,
        #        "epsilon": 0.01,
        #    }
        #]
        aggregations: list[dict[str, Any]],
        #[
        #    {
        #        "lead_time": 0,
        #        "span": 1,
        #        "agg": "SUM",
        #    },
        #]
        force_integer_forecasts: bool = False,
    ):
        self.horizon = base.horizon
        self.seasonality = base.seasonality
        self.num_windows = base.num_windows
        self._future_series_per_window = _extract_future_target_from(task=base)
        self.aggregations = aggregations
        self.metrics = metrics

    def evaluation_summary(
        self,
        y_pred_per_window, # [N, S, H, D]
    ):
        y_true_per_window = self._future_series_per_window
        # TODO: Ensure that S of y_true matches S of y_pred
        evals = []
        for w in range(self.num_windows):
            for a in self.aggregations:
                y_true = _aggregate_out_horizon(
                    y=y_true_per_window[w],
                    lead_time=a["lead_time"],
                    span=a["span"],
                    agg_function=a["agg"]
                )
                y_pred = _aggregate_out_horizon(
                    y=y_pred_per_window[w],
                    lead_time=a["lead_time"],
                    span=a["span"],
                    agg_function=a["agg"]
                )
                for metric_config in self.metrics:
                    metric = get_metric(metric=metric_config)
                    value = metric.compute(
                        y_true=y_true,
                        y_pred=y_pred,
                        reduce_axes=(0,1),
                    )
                    evals.append(
                        {
                            "window": w,
                            "aggregation": a,
                            "metric": metric.name,
                            "value": value,
                        }
                    )
        return evals

def _extract_future_target_from(task: FevTask):
    future_series_per_window = []
    for window in task.iter_windows():
        # As used in `window._prepare_arrays()` in `Task.evaluation_summary()`
        # https://github.com/autogluon/fev/blob/f9e2c7d3152ed7d15c3b1c2877bb3ae7d977046f/src/fev/task.py#L130
        _, _, future_data = window._get_past_future_test_data()
        future_series = [ts[task.target] for ts in future_data] # List of numpy arrays, one list item per time series
        future_series = np.stack([np.tile(x, reps=2501).reshape(2501, task.horizon) for x in future_series])
        future_series = future_series[:, :, :, None] # TODO: Adjust for multiple target
        future_series_per_window.append(future_series)
    return future_series_per_window

def _aggregate_out_horizon(
    y: np.ndarray, # (N, S, H, D)
    lead_time: int,
    span: int,
    agg_function: str,
    keepdims=True,
) -> np.ndarray: # (N, S, 1, D) if keepdims else (N, S, D)
    return AVAILABLE_AGGREGATION_FUNCTIONS[agg_function](
        a=y[:, :, lead_time:(lead_time+span), :], # Take subset of horizon
        axis=2, # Assumes that horizon is the third dimension
        keepdims=keepdims
    )

AVAILABLE_AGGREGATION_FUNCTIONS = {
    "MIN": np.min,
    "MAX": np.max,
    "MEAN": np.mean,
    "SUM": np.sum,
}

# Add more aggregations, such as:
# - Each gt / lt
