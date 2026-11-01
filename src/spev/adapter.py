import numpy as np
import fev
import shuffler as sf

from datasets import DatasetDict

def reshape_fev_predictions_per_window_to_sample_paths(
    predictions_per_window: list[DatasetDict],
    task: fev.Task,
    n_samples: int = 2501,
    force_marginal: bool = False,
) -> list[np.array]:
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
        # As used in `window._prepare_arrays()` in `Task.evaluation_summary()`
        # https://github.com/autogluon/fev/blob/f9e2c7d3152ed7d15c3b1c2877bb3ae7d977046f/src/fev/task.py#L130
        past_data, _ = window.get_input_data()
        past_series = [ts[task.target] for ts in past_data] # List of numpy arrays, one list item per time series
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
