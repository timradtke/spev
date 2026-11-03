import numpy as np

from datasets import Dataset
from fev import Task
from typing import Any
from .aggregates import AggregateConfig, get_aggregate
from .metrics import MetricConfig, get_metric

# TODO: Consider creating SamplePathTask as subclass of Task. Not ideal given
#       the different kinds of metrics?
class SamplePathTask:
    """
    A sample path evaluation task.
    """
    def __init__(
        self,
        base: Task,
        metrics: list[MetricConfig],
        aggregates: list[AggregateConfig],
        force_integer_forecasts: bool = False,
    ):
        """
        Initialize a SamplePathTask.

        Parameters
        ----------
        base : Task
            The Task that serves as basis of the new SamplePathTask. The future
            target that is unknown at prediction time but needed for evaluation
            will be extracted via the Task when you initialize the
            SamplePathTask. This might involve downloads from HuggingFace or
            other locations. The SamplePathTask object does not store any other
            data.
        metrics : list[MetricConfig]
            The metrics based on which predictions for this task will be
            evaluated. Metrics are configured as either a string (e.g., "WCRPS")
            or as dictionary to pass additional parameters. The quantile loss
            is defined with a `quantile_level`, for example:
            `{"name": "WQL", "quantile_level": 0.9}`
        aggregates : list[AggregateConfig]
            Aggregates define how to aggregate the forecast horizon before
            evaluation of sample paths. Each aggregate needs to be defined based
            on a `lead_time` and `span`.
            For example, when forecasting the next 8 months of Car Parts, you
            could choose the aggregate
            `{"name": "SUM", "lead_time": 2, "span": 4}` to sum the 4 monthly
            predictions following the lead time of 2 months.
            In contrast, when evaluating a prediction of peak load over a
            given span you could choose
            `{"name": "MAX", "lead_time": 12, "span": 24}`.
        force_integer_forecasts : bool
            Not implemented, ignored.
        """
        self.horizon = base.horizon
        self.seasonality = base.seasonality
        self.num_windows = base.num_windows
        self._future_series_per_window = _extract_future_target_from(task=base)
        self.aggregates = aggregates
        self.metrics = metrics

    def evaluation_summary(
        self,
        y_pred_per_window: list[np.ndarray], # [N, S, H, D]
    ) -> list[dict[Any]]:
        """
        Evaluate predicted sample paths on the task at hand.

        Parameters
        ----------
        y_pred_per_window : list[np.ndarray] (N, S, H, D)
            The predictions to evaluate on the task. A list of four-dimensional
            arrays. The first dimension's size equals the number of time series,
            the second is the number of sample paths, the third is the forecast
            horizon, and the fourth is the number of targets.

        Returns
        -------
        list[dict[Any]]
            The evaluation for each combination of the task's metrics,
            aggregates, and windows.
        """

        y_true_per_window = self._future_series_per_window
        evals = []
        for w in range(self.num_windows):
            for a in self.aggregates:
                aggregate = get_aggregate(aggregate=a)
                y_true = aggregate.compute(
                    a=y_true_per_window[w],
                    axis=2,
                    keepdims=True,
                )
                y_pred = aggregate.compute(
                    a=y_pred_per_window[w],
                    axis=2,
                    keepdims=True,
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
                            "aggregate": aggregate.name,
                            "metric": metric.name,
                            "value": value,
                        }
                    )
        return evals

def _extract_future_target_from(
    task: Task
) -> list[np.ndarray]: # List of arrays of shape (N, 1, H, D)
    future_series_per_window = []
    for window in task.iter_windows():
        # As used in `window._prepare_arrays()` in `Task.evaluation_summary()`.
        # Problematic as it's a private method...
        # https://github.com/autogluon/fev/blob/f9e2c7d3152ed7d15c3b1c2877bb3ae7d977046f/src/fev/task.py#L130
        _, _, future_data = window._get_past_future_test_data()
        future_series = _stack_series_from_dataset_into_arrays_per_window(
            future_data=future_data,
            target=task.target,
        )
        future_series_per_window.append(future_series)
    return future_series_per_window

def _stack_series_from_dataset_into_arrays_per_window(
    future_data: Dataset | list[dict[np.ndarray]],
    target: str
) -> np.ndarray: # (N, 1, H, D) to broadcast to (N, S, H, D) of predictions.
    """
    Stack individual series arrays of dimensions (H, D) into an array of all
    series with shape (N, 1, H, D).
    """
    # List of numpy arrays, one list item per time series
    # TODO: Adjust for multiple targets
    future_series = np.stack([ts[target] for ts in future_data])
    future_series = future_series[:, None, :, None]
    return future_series
