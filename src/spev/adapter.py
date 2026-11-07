import numpy as np

from datasets import DatasetDict
from fev import Task
from . import _HAS_SHUFFLER

if not _HAS_SHUFFLER:
    raise ImportError(
        """Install the optional `spev[adapter]` dependencies to use `spev.adapter`."""
    )
import shuffler as sf

def reshape_fev_predictions_to_sample_paths(
    predictions_per_window: list[DatasetDict],
    task: Task,
    n_samples: int = 2501,
    force_marginal: bool = False,
) -> list[np.ndarray]:
    """
    Transform quantile predictions generated for fev into sample paths.

    Takes quantile predictions generated for evaluation via a fev Task, pulls
    past actuals of the data set from the fev Task, and combines both to
    generate sample path forecasts using Shuffler that can be evaluated by a
    SamplePathTask.

    Use this function to evaluate models that can only generate marginal
    quantile forecasts on tasks that require sample paths. The quality of the
    evaluated sample paths will depend both on the original quantile forecasts
    as well as their transformation into sample paths via Shuffler.

    Parameters
    ----------
    predictions_per_window : list[DatasetDict]
        Predictions prepared in the shape that a fev Task expects for
        evaluation, see `models.Chronos2Model`.
    task : Task
        The fev Task for which the predictions were generated. Is used as
        provider of the training data that Shuffler fits copulas against to
        generate sample paths.
    n_samples : int
        The number of sample paths to draw.
    force_marginal : bool
        If True, draw sample paths by sampling from the original marginal
        forecast distributions instead of from the Shuffler-approximated
        joint forecast distribution.
    
    Returns
    -------
    y_pred_per_window : list[np.ndarray] (N, S, H, D)
        A list of four-dimensional arrays of sample path forecasts. Each element
        of the list contains the forecasts for one of the task's forecast
        windows. This is the shape expected by
        `SamplePathTask.evaluation_summary()`.
        The first dimension's size equals the number of time series, the second
        is the number of sample paths, the third is the forecast horizon, and
        the fourth is the number of targets.
    """
    windows_of_ls_quantile_array_prediction_per_series = []
    for window in predictions_per_window:
        ls_quantile_array_prediction_per_series = []
        
        # TODO: Extend for multiple targets
        for quantile_dict in window["target"]:
            preds = np.column_stack([quantile_dict[str(q)] for q in task.quantile_levels]).transpose()
            ls_quantile_array_prediction_per_series.append(preds)
        
        windows_of_ls_quantile_array_prediction_per_series.append(
            ls_quantile_array_prediction_per_series
        )

    past_series_per_window = []
    for window in task.iter_windows():
        past_data, _ = window.get_input_data()
        # List of numpy arrays, one item per time series:
        past_series = [ts[task.target] for ts in past_data]
        past_series_per_window.append(past_series)

    windows_of_ls_sample_paths_per_series = []
    for window_idx in range(task.num_windows):
        ls_sample_paths_per_series = []
        past_series = past_series_per_window[window_idx]
        ls_quantile_array_prediction_per_series = windows_of_ls_quantile_array_prediction_per_series[window_idx]

        for ts_idx in range(len(ls_quantile_array_prediction_per_series)):
            y = np.nan_to_num(past_series[ts_idx], nan=0.0)
            sample_paths = sf.draw_sample_paths(
                quantile_forecasts=ls_quantile_array_prediction_per_series[ts_idx],
                quantile_forecast_levels=np.array(task.quantile_levels),
                y=(None if force_marginal else y),
                horizon=task.horizon,
                size=n_samples,
                seed=ts_idx,
            )

            ls_sample_paths_per_series.append(sample_paths)
        
        windows_of_ls_sample_paths_per_series.append(ls_sample_paths_per_series)

    y_pred_per_window = []
    for window_idx in range(task.num_windows):
        y_pred_per_window.append(
            np.stack(windows_of_ls_sample_paths_per_series[window_idx])[:, :, :, None]
        )

    return y_pred_per_window
