# SPEV

**S**ample **P**ath **EV**aluation library, inspired by [FEV](https://github.com/autogluon/fev/tree/main). It is work in progress.

FEV-bench is a fantastic forecast benchmark and the accompanying `fev` library makes it comfortable to compare models on a set of predefined tasks.

However, like most forecast benchmarks, FEV focuses its evaluation on point forecasts and quantiles of marginal distribution forecasts. SPEV is a proof-of-concept of how forecast tasks can be extended to evaluate the joint forecast distribution over the forecast horizon.

At the moment, SPEV defines its sample path tasks as a combination of aggregates and metrics to implement the kind of evaluation of joint forecast distributions in the form of sample paths as performed in, for example, [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110) (Salinas et al., 2017), sections 4.1 and 6.

## Install SPEV

Install spev from [Github](https://github.com/timradtke/spev):

```bash
uv pip install "git+https://github.com/timradtke/spev"
```

## Get Started

Let's get you started with a minimal example. For an extended tutorial including additional aggregates and metrics, as well as evaluation of [Chronos-2](https://github.com/amazon-science/chronos-forecasting) marginal forecasts adapted to sample paths using [Shuffler](https://github.com/timradtke/shuffler), refer to [this notebook](https://github.com/timradtke/spev/blob/main/docs/evaluate_car_parts_using_spev.ipynb).

Import spev and fev as a FEV task serves as base for a SPEV task.

```python
import numpy as np

from fev import Task
from spev import SamplePathTask
```

### Define the FEV Task

The FEV task is the basis for the evaluation. The SPEV sample path task only extends it to evaluate the same task but with sample path evaluation metrics.

Forecast the last eight months of the car parts dataset.

```python
base_task = Task(
    dataset_path="autogluon/chronos_datasets",
    dataset_config="monash_car_parts",
    horizon=8,
    num_windows=1,
    seasonality=12,
    target="target",
)
```

### Define the SPEV Task

The sample path task extends the base task by defining aggregates of sample paths, and metrics calculated on the aggregated sample paths. The `SUM` aggregate and the quantile loss metrics implement the evaluation applied onto the car parts dataset in the aforementioned Salinas et al. (2017). Quoting from their section 4.1:

> The metrics are evaluated for a certain *spans* [L,L+S) in the prediction range, where L is a *lead time* after the forecast start point.

Aggregates in SPEV implement the same approach of using lead time and span to define subsets of the forecast horizon that need to be aggregated for evaluation.

```python
# Initialization of the `SamplePathTask` will download the dataset
# defined in the `Task` (in this case from HuggingFace Hub).

task = SamplePathTask(
    base=base_task,
    aggregates=[
        {"name": "SUM", "lead_time": 0, "span": 1},
        {"name": "SUM", "lead_time": 2, "span": 1},
        {"name": "SUM", "lead_time": 0, "span": 8},
    ],
    metrics=[
        {"name": "WQL", "quantile_level": 0.9, "epsilon": 0.01},
        {"name": "WQL", "quantile_level": 0.5, "epsilon": 0.01},
    ]
)
```

### Evaluate Using SPEV Task

With the task defined, you can evaluate sample path forecasts on the task. For simplicity, draw sample paths by sampling from a Poisson distribution. The sample path forecasts need to be a list with one array per forecast window defined in the FEV task (in this case, one). Each array is four-dimensional: The first dimension is the number of time series, the second dimension is the number of samples, the third dimension is the length of the forecast horizon, and the fourth is the number of targets. In this case, the shape will be (2674, 2501, 8, 1).

```python
rng = np.random.default_rng(seed=512)
sample_path_forecasts = [
    rng.poisson(
        lam=0.25,
        size=(
            2674, # Number of time series
            2501, # Number of sample paths
            8,    # Forecast horizon length
            1,    # Number of targets
        )
    )
]
```

Pass the sample path forecasts to the SPEV task for evaluation. It returns a list of evaluated metrics, one for each combination of aggregates and metrics defined in the sample path task. Here, with 3 aggregates and 2 metrics, there will be 6 evaluated metrics.

```python
task.evaluation_summary(
    y_pred_per_window=sample_path_forecasts
)
```
```
[{'window': 0,
  'aggregate': 'Sum_[0,1]',
  'metric': 'WeightedQuantileLoss@0.9',
  'value': array([2.9842507])},
 {'window': 0,
  'aggregate': 'Sum_[0,1]',
  'metric': 'WeightedQuantileLoss@0.5',
  'value': array([1.00000002])},
 {'window': 0,
  'aggregate': 'Sum_[2,1]',
  'metric': 'WeightedQuantileLoss@0.9',
  'value': array([3.03936339])},
 {'window': 0,
  'aggregate': 'Sum_[2,1]',
  'metric': 'WeightedQuantileLoss@0.5',
  'value': array([0.99999997])},
 {'window': 0,
  'aggregate': 'Sum_[0,8]',
  'metric': 'WeightedQuantileLoss@0.9',
  'value': array([1.31726945])},
 {'window': 0,
  'aggregate': 'Sum_[0,8]',
  'metric': 'WeightedQuantileLoss@0.5',
  'value': array([0.893785])}]
```

For the most part, predicting a constant zero would have been better:

```python
task.evaluation_summary(
    y_pred_per_window=[np.zeros(shape=(2674, 2501, 8, 1))],
)
```
```
[{'window': 0,
  'aggregate': 'Sum_[0,1]',
  'metric': 'WeightedQuantileLoss@0.9',
  'value': array([0.2])},
 {'window': 0,
  'aggregate': 'Sum_[0,1]',
  'metric': 'WeightedQuantileLoss@0.5',
  'value': array([1.00000002])},
 {'window': 0,
  'aggregate': 'Sum_[2,1]',
  'metric': 'WeightedQuantileLoss@0.9',
  'value': array([0.19999999])},
 {'window': 0,
  'aggregate': 'Sum_[2,1]',
  'metric': 'WeightedQuantileLoss@0.5',
  'value': array([0.99999997])},
 {'window': 0,
  'aggregate': 'Sum_[0,8]',
  'metric': 'WeightedQuantileLoss@0.9',
  'value': array([0.19999999])},
 {'window': 0,
  'aggregate': 'Sum_[0,8]',
  'metric': 'WeightedQuantileLoss@0.5',
  'value': array([0.99999997])}]
  ```

## References

Ansari et al. (2025): Chronos-2: From Univariate to Universal Forecasting. https://arxiv.org/abs/2510.15821 and https://github.com/amazon-science/chronos-forecasting.

Salinas et al. (2017): DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks. https://arxiv.org/abs/1704.04110.

Seeger et al. (2016): Bayesian Intermittent Demand Forecasting for Large Inventories. https://proceedings.neurips.cc/paper/2016/hash/03255088ed63354a54e0e5ed957e9008-Abstract.html

Seeger et al. (2017): Approximate Bayesian Inference in Linear State Space Models for Intermittent Demand Forecasting at Scale. https://arxiv.org/abs/1709.07638

Shchur et al. (2025): fev-bench: A Realistic Benchmark for Time Series Forecasting. https://arxiv.org/abs/2509.26468 and https://github.com/autogluon/fev.
