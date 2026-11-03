from typing import Any
import numpy as np

class Aggregate:
    """Base class for all aggregates."""
    def __init__(self, lead_time: int, span: int):
        self.lead_time = lead_time
        self.span = span

    @property
    def name(self) -> str:
        """Name of the aggregation."""
        return (
            self.__class__.__name__
            + "_[" + str(self.lead_time) + "," + str(self.span) + "]"
        )

    def slice_horizon(
        self,
        a: np.ndarray, # Expected to be of shape (N, S, H, D)
    ) -> np.ndarray:
        """
        Slice the provided array's horizon to the span following the lead time.

        Parameters
        ----------
        a : np.ndarray (N, S, H, D)
            Array of observations or predictions. N=number of series,
            S=number of sample paths, H=forecast horizon, D=target dimensions.
        
        Returns
        -------
        np.ndarray (N, S, span, D)
            Sliced version of input array.
        """
        return a[:, :, self.lead_time:(self.lead_time+self.span), :]

    def compute(
        self,
        a: np.ndarray, # Expected to be of shape (N, S, H, D)
        axis: int,
        keepdims: bool,
    ) -> np.ndarray:
        """Aggregate of array elements over a given axis.

        Parameters
        ----------
        a : np.ndarray (N, S, H, D)
            Array of observations or predictions. N=number of series,
            S=number of sample paths, H=forecast horizon, D=target dimensions.
        axis : int
            Axis along which the aggregation is performed.
        keepdims : bool
            The aggregated dimension will be left in the returned array with
            size one.
        
        Returns
        -------
        np.ndarray
            Aggregate of input array, with specified axes removed.
        """
        raise NotImplementedError

class Sum(Aggregate):
    def compute(
        self,
        a: np.ndarray,
        axis: int,
        keepdims: bool,
    ) -> np.ndarray:
        return np.sum(a = self.slice_horizon(a), axis=axis, keepdims=keepdims)

class Mean(Aggregate):
    def compute(
        self,
        a: np.ndarray,
        axis: int,
        keepdims: bool,
    ) -> np.ndarray:
        return np.mean(a = self.slice_horizon(a), axis=axis, keepdims=keepdims)

class Min(Aggregate):
    def compute(
        self,
        a: np.ndarray,
        axis: int,
        keepdims: bool,
    ) -> np.ndarray:
        return np.min(a = self.slice_horizon(a), axis=axis, keepdims=keepdims)

class Max(Aggregate):
    def compute(
        self,
        a: np.ndarray,
        axis: int,
        keepdims: bool,
    ) -> np.ndarray:
        return np.max(a = self.slice_horizon(a), axis=axis, keepdims=keepdims)

class AnyGreaterThan(Aggregate):
    def __init__(self, lead_time: int, span: int, k: float = -np.inf) -> None:
        super().__init__(lead_time=lead_time, span=span)
        self.k = k

    @property
    def name(self) -> str:
        return super().name + "_x>" + str(self.k)

    def compute(
        self,
        a: np.ndarray,
        axis: int,
        keepdims: bool,
    ):
        return (
            np.any(
                a=(self.slice_horizon(a=a) > self.k),
                axis=axis,
                keepdims=keepdims
            )
            .astype(float)
        )

class AnyLessThan(Aggregate):
    def __init__(self, lead_time: int, span: int, k: float = np.inf) -> None:
        super().__init__(lead_time=lead_time, span=span)
        self.k = k

    @property
    def name(self) -> str:
        return super().name + "_x<" + str(self.k)

    def compute(
        self,
        a: np.ndarray,
        axis: int,
        keepdims: bool,
    ):
        return (
            np.any(
                a=(self.slice_horizon(a=a) < self.k),
                axis=axis,
                keepdims=keepdims
            )
            .astype(float)
        )

class AnyBetween(Aggregate):
    def __init__(
        self,
        lead_time: int,
        span: int,
        lb: float = -np.inf,
        ub: float = np.inf
    ) -> None:
        super().__init__(lead_time=lead_time, span=span)
        self.lb = lb
        self.ub = ub
        
    @property
    def name(self) -> str:
        return super().name + "_(" + str(self.lb) + "<x<" + str(self.ub) + ")"

    def compute(
        self,
        a: np.ndarray,
        axis: int,
        keepdims: bool,
    ):
        a_subset = self.slice_horizon(a=a)
        return (
            np.any(
                a=((a_subset > self.lb) & (a_subset < self.ub)),
                axis=axis,
                keepdims=keepdims
            )
            .astype(float)
        )

class AllGreaterThan(Aggregate):
    def __init__(self, lead_time: int, span: int, k: float = -np.inf) -> None:
        super().__init__(lead_time=lead_time, span=span)
        self.k = k

    @property
    def name(self) -> str:
        return super().name + "_x>" + str(self.k)

    def compute(
        self,
        a: np.ndarray,
        axis: int,
        keepdims: bool,
    ):
        return (
            np.all(
                a=(self.slice_horizon(a=a) > self.k),
                axis=axis,
                keepdims=keepdims
            )
            .astype(float)
        )

class AllLessThan(Aggregate):
    def __init__(self, lead_time: int, span: int, k: float = np.inf) -> None:
        super().__init__(lead_time=lead_time, span=span)
        self.k = k

    @property
    def name(self) -> str:
        return super().name + "_x<" + str(self.k)

    def compute(
        self,
        a: np.ndarray,
        axis: int,
        keepdims: bool,
    ):
        return (
            np.all(
                a=(self.slice_horizon(a=a) < self.k),
                axis=axis,
                keepdims=keepdims
            )
            .astype(float)
        )

class AllBetween(Aggregate):
    def __init__(
        self,
        lead_time: int,
        span: int,
        lb: float = -np.inf,
        ub: float = np.inf
    ) -> None:
        super().__init__(lead_time=lead_time, span=span)
        self.lb = lb
        self.ub = ub
        
    @property
    def name(self) -> str:
        return super().name + "_(" + str(self.lb) + "<x<" + str(self.ub) + ")"

    def compute(
        self,
        a: np.ndarray,
        axis: int,
        keepdims: bool,
    ):
        a_subset = self.slice_horizon(a=a)
        return (
            np.all(
                a=((a_subset > self.lb) & (a_subset < self.ub)),
                axis=axis,
                keepdims=keepdims
            )
            .astype(float)
        )

class CountGreaterThan(Aggregate):
    def __init__(self, lead_time: int, span: int, k: float = -np.inf) -> None:
        super().__init__(lead_time=lead_time, span=span)
        self.k = k

    @property
    def name(self) -> str:
        return super().name + "_x>" + str(self.k)

    def compute(
        self,
        a: np.ndarray,
        axis: int,
        keepdims: bool,
    ):
        return (
            np.sum(
                a=(self.slice_horizon(a=a) > self.k),
                axis=axis,
                keepdims=keepdims
            )
        )

class CountLessThan(Aggregate):
    def __init__(self, lead_time: int, span: int, k: float = np.inf) -> None:
        super().__init__(lead_time=lead_time, span=span)
        self.k = k

    @property
    def name(self) -> str:
        return super().name + "_x<" + str(self.k)

    def compute(
        self,
        a: np.ndarray,
        axis: int,
        keepdims: bool,
    ):
        return (
            np.sum(
                a=(self.slice_horizon(a=a) < self.k),
                axis=axis,
                keepdims=keepdims
            )
        )

class CountBetween(Aggregate):
    def __init__(
        self,
        lead_time: int,
        span: int,
        lb: float = -np.inf,
        ub: float = np.inf
    ) -> None:
        super().__init__(lead_time=lead_time, span=span)
        self.lb = lb
        self.ub = ub
        
    @property
    def name(self) -> str:
        return super().name + "_(" + str(self.lb) + "<x<" + str(self.ub) + ")"

    def compute(
        self,
        a: np.ndarray,
        axis: int,
        keepdims: bool,
    ):
        a_subset = self.slice_horizon(a=a)
        return (
            np.sum(
                a=((a_subset > self.lb) & (a_subset < self.ub)),
                axis=axis,
                keepdims=keepdims
            )
        )

# In contrast to MetricConfig this has to be a dict due to lead_time and span.
AggregateConfig = dict[str, Any]

def get_aggregate(aggregate: AggregateConfig) -> Aggregate:
    """Get an aggregate class by name or configuration."""
    aggregate_name = aggregate["name"]
    try:
        aggregate_type = AVAILABLE_AGGREGATES[aggregate_name.upper()]
    except KeyError:
        raise ValueError(
            f"Aggregation method '{aggregate_name}' is not available. Available aggregates: {sorted(AVAILABLE_AGGREGATES)}"
        )

    if isinstance(aggregate, dict):
        return aggregate_type(
            **{k: v for k, v in aggregate.items() if k != "name"}
        )
    else:
        raise ValueError(f"Invalid metric configuration: {aggregate}")

AVAILABLE_AGGREGATES = {
    "MIN": Min,
    "MAX": Max,
    "MEAN": Mean,
    "SUM": Sum,
    "ANY_GT": AnyGreaterThan,
    "ANY_LT": AnyLessThan,
    "ANY_BETWEEN": AnyBetween,
    "ALL_GT": AllGreaterThan,
    "ALL_LT": AllLessThan,
    "ALL_BETWEEN": AllBetween,
    "COUNT_GT": CountGreaterThan,
    "COUNT_LT": CountLessThan,
    "COUNT_BETWEEN": CountBetween,
}
