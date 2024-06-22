from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from cfair.backends import NumpyBackend, TorchBackend, Backend, TensorflowBackend


class Metric:
    """Interface of a fairness metric for continuous attributes."""

    @dataclass(frozen=True, init=True, repr=False, eq=False, unsafe_hash=None)
    class Result:
        """Data class representing the results of a metric computation."""

        a: Any = field()
        """The first of the two vectors on which the metric is computed."""

        b: Any = field()
        """The first of the two vectors on which the metric is computed."""

        value: Any = field()
        """The value measured by the metric, optionally with gradient information attached."""

        metric: Any = field()
        """The metric instance that generated this result."""

        num_call: int = field()
        """The n-th time at which the metric instance that generated the result was called."""

    def __init__(self, backend: Union[str, Backend]):
        """
        :param backend:
            The backend to use to compute the metric, or its alias.
        """
        if backend == 'numpy':
            backend = NumpyBackend()
        elif backend == 'tensorflow':
            backend = TensorflowBackend()
        elif backend == 'torch':
            backend = TorchBackend()
        else:
            assert isinstance(backend, Backend), f"Unknown backend '{backend}'"
        self._backend: Backend = backend
        self._last_result: Optional[Metric.Result] = None
        self._num_calls: int = 0

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def last_result(self) -> Optional[Result]:
        """The `Result` instance returned from the last metric call, or None if no call was performed."""
        return self._last_result

    @property
    def num_calls(self) -> int:
        """The number of times that this metric instance was called."""
        return self._num_calls

    def value(self, a, b) -> Any:
        """Computes the metric.

        :param a:
            The first vector.

        :param b:
            The second vector.

        :result:
            A scalar value representing the computed metric value, optionally with gradient information attached."""
        return self(a=a, b=b).value

    def __call__(self, a, b) -> Result:
        """Computes the metric.

        :param a:
            The first vector.

        :param b:
            The second vector.

        :result:
            A `Result` instance containing the computed metric value together with additional information.
        """
        bk = self.backend
        assert bk.ndim(a) == bk.ndim(b) == 1, f"Expected vectors with one dimension, got {bk.ndim(a)} and {bk.ndim(b)}"
        assert bk.len(a) == bk.len(b), f"Input vectors must have the same dimension, got {bk.len(a)} != {bk.len(b)}"
        self._num_calls += 1
        res = self._compute(a=bk.cast(a, dtype=float), b=bk.cast(b, dtype=float))
        self._last_result = res
        return res

    @abstractmethod
    def _compute(self, a, b) -> Result:
        pass


class CopulaMetric(Metric):
    """Interface of a fairness metric for continuous attributes using copula transformations."""

    def f(self, a) -> Any:
        """Returns the mapped vector f(a) using the copula transformation f computed in the last execution.

        :param a:
            The vector to be projected.

        :return:
            The resulting projection.
        """
        assert self.last_result is not None, "The metric has not been computed yet, so no transformation can be used."
        return self._f(a=a)

    def g(self, b) -> Any:
        """Returns the mapped vector g(b) using the copula transformation g computed in the last execution.

        :param b:
            The vector to be projected.

        :return:
            The resulting projection.
        """
        assert self.last_result is not None, "The metric has not been computed yet, so no transformation can be used."
        return self._g(b=b)

    @abstractmethod
    def _f(self, a) -> Any:
        pass

    @abstractmethod
    def _g(self, b) -> Any:
        pass
