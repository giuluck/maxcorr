from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from cfair.backends import NumpyBackend, TorchBackend, Backend, TensorflowBackend


class Indicator:
    """Interface of a fairness indicator for continuous attributes."""

    @dataclass(frozen=True, init=True, repr=False, eq=False)
    class Result:
        """Data class representing the results of an indicator computation."""

        a: Any = field()
        """The first of the two vectors on which the indicator is computed."""

        b: Any = field()
        """The first of the two vectors on which the indicator is computed."""

        value: Any = field()
        """The value measured by the indicator, optionally with gradient information attached."""

        indicator: Any = field()
        """The indicator instance that generated this result."""

        num_call: int = field()
        """The n-th time at which the indicator instance that generated the result was called."""

    def __init__(self, backend: Union[str, Backend]):
        """
        :param backend:
            The backend to use to compute the indicator, or its alias.
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
        self._last_result: Optional[Indicator.Result] = None
        self._num_calls: int = 0

    @property
    def backend(self) -> Backend:
        """The backend to use to compute the indicator."""
        return self._backend

    @property
    def last_result(self) -> Optional[Result]:
        """The `Result` instance returned from the last indicator call, or None if no call was performed."""
        return self._last_result

    @property
    def num_calls(self) -> int:
        """The number of times that this indicator instance was called."""
        return self._num_calls

    def compute(self, a, b) -> Any:
        """Computes the indicator.

        :param a:
            The first vector.

        :param b:
            The second vector.

        :result:
            A scalar value representing the computed indicator value, optionally with gradient information attached."""
        return self(a=a, b=b).value

    def __call__(self, a, b) -> Any:
        """Computes the indicator.

        :param a:
            The first vector.

        :param b:
            The second vector.

        :result:
            A `Result` instance containing the computed indicator value together with additional information.
        """
        bk = self.backend
        assert bk.ndim(a) == bk.ndim(b) == 1, f"Expected vectors with one dimension, got {bk.ndim(a)} and {bk.ndim(b)}"
        assert bk.len(a) == bk.len(b), f"Input vectors must have the same dimension, got {bk.len(a)} != {bk.len(b)}"
        self._num_calls += 1
        res = self._compute(a=bk.cast(a, dtype=float), b=bk.cast(b, dtype=float))
        self._last_result = res
        return res

    @abstractmethod
    def _factor(self, a, b) -> Any:
        """The scaling factor to compute the indicator based on its semantics.

        :param a:
            The first vector.

        :param b:
            The second vector.

        :result:
            A value respecting the backend types representing the scaling factor.
        """
        pass

    @abstractmethod
    def _compute(self, a, b) -> Result:
        """Computes the indicator value without performing any additional check.

        :param a:
            The first vector.

        :param b:
            The second vector.

        :result:
            A `Result` instance containing the computed indicator value together with additional information.
        """
        pass


class CopulaIndicator(Indicator):
    """Interface of a fairness indicator for continuous attributes using copula transformations."""

    def __init__(self, backend: Union[str, Backend], eps: float):
        """
        :param backend:
            The backend to use to compute the indicator, or its alias.

        :param eps:
            The epsilon value used to avoid division by zero in case of null standard deviation.
        """
        super(CopulaIndicator, self).__init__(backend=backend)
        self._eps: float = eps

    @property
    def eps(self) -> float:
        """The epsilon value used to avoid division by zero in case of null standard deviation."""
        return self._eps

    @abstractmethod
    def _f(self, a) -> Any:
        pass

    @abstractmethod
    def _g(self, b) -> Any:
        pass

    def f(self, a) -> Any:
        """Returns the mapped vector f(a) using the copula transformation f computed in the last execution.

        :param a:
            The vector to be projected.

        :return:
            The resulting projection with zero mean and unitary variance.
        """
        assert self.last_result is not None, "The indicator has not been computed yet, no transformation can be used."
        return self._f(a)

    def g(self, b) -> Any:
        """Returns the mapped vector g(b) using the copula transformation g computed in the last execution.

        :param b:
            The vector to be projected.

        :return:
            The resulting projection with zero mean and unitary variance.
        """
        assert self.last_result is not None, "The indicator has not been computed yet, no transformation can be used."
        return self._g(b)

    def value(self, a, b) -> float:
        """Gets the indicator value using the stored copula transformations on the two given vectors.

        :param a:
            The first vector.

        :param b:
            The second vector.

        :return:
            The computed indicator.
        """
        assert self.last_result is not None, "The indicator has not been computed yet, no transformation can be used."
        fa = self.backend.standardize(self._f(a=a), eps=self.eps)
        gb = self.backend.standardize(self._g(b=b), eps=self.eps)
        value = self.backend.mean(fa * gb) * self._factor(a=a, b=b)
        return self.backend.item(value)


class HGRIndicator(Indicator, ABC):
    """Hirschfield-Gebelin-Renyi indicator (i.e., Non-Linear Pearson's correlation).

    The first mapped vector is:
        x = fa * std(a), where mean(fa) = 0 and std(fa) = 1
    and the second mapped vector is:
        y = gb * std(b), where mean(gb) = 0 and std(gb) = 1
    i.e., both mapped vectors (x, y) are centered and with standard deviation equal to the original vectors (a, b).

    HGR is eventually computed as:
        pearson(x, y) = cov(x, y) / std(x) / std(y) =
                      = cov(x, y) / std(a) / std(b) =
                      = cov(fa * std(a), gb * std(b)) / std(a) / std(b) =
                      = cov(fa, gb) =
                      = mean(fa * gb)
    i.e., HGR is the average of the product between the standardized copula transformations (without a scaling factor).
    """

    def _factor(self, a, b) -> Any:
        return 1


class GeDIIndicator(Indicator, ABC):
    """Generalized Disparate Impact indicator.

    The first mapped vector is:
        x = fa * std(a), where mean(fa) = 0 and std(fa) = 1
    and the second mapped vector is:
        y = gb * std(b), where mean(gb) = 0 and std(gb) = 1
    i.e., both mapped vectors (x, y) are centered and with standard deviation equal to the original vectors (a, b).

    GeDI is eventually computed as:
        cov(x, y) / var(x) = cov(x, y) / var(a) =
                           = cov(x, y) / std(a) / std(a) =
                           = cov(fa * std(a), gb * std(b)) / std(a) / std(a) =
                           = cov(fa, gb) * std(b) / std(a) =
                           = mean(fa * gb) * std(b) / std(a)
    i.e., GeDI is the average of the product between the standardized copula transformations (HGR) multiplied by a
    scaling factor std(b) / std(a).
    """

    def _factor(self, a, b) -> Any:
        return self.backend.std(b) / self.backend.std(a)


class NLCIndicator(Indicator, ABC):
    """Non-Linear Covariance (NLC) computed using two neural networks to approximate the copula transformations.

    The first mapped vector is:
        x = fa * std(a), where mean(fa) = 0 and std(fa) = 1
    and the second mapped vector is:
        y = gb * std(b), where mean(gb) = 0 and std(gb) = 1
    i.e., both mapped vectors (x, y) are centered and with standard deviation equal to the original vectors (a, b).

    NLC is eventually computed as:
        cov(x, y) = cov(x, y) =
                  = cov(fa * std(a), gb * std(b)) =
                  = cov(fa, gb) * std(b) * std(a) =
                  = mean(fa * gb) * std(b) * std(a)
    i.e., NLC is the average of the product between the standardized copula transformations (HGR) multiplied by a
    scaling factor std(b) * std(a).
    """

    def _factor(self, a, b) -> Any:
        return self.backend.std(b) * self.backend.std(a)
