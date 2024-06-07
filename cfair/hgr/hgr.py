from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from cfair.backend import NumpyBackend, TorchBackend, Backend


@dataclass(frozen=True, init=True, repr=False, eq=False, unsafe_hash=None)
class HGR:
    """Interface for an object that computes the HGR correlation."""

    @dataclass(frozen=True, init=True, repr=False, eq=False, unsafe_hash=None)
    class Result:
        """Data class representing the results of an HGR computation."""

        a: Any = field()
        """The first of the two vectors on which the HGR correlation is computed."""

        b: Any = field()
        """The first of the two vectors on which the HGR correlation is computed."""

        correlation: Any = field()
        """The actual value of the correlation, optionally with gradient information attached."""

        hgr: Any = field()
        """The HGR instance that generated this result."""

        num_call: int = field()
        """The n-th time at which the HGR instance that generated the result was called."""

    class _State:
        """Data class representing the internal mutable state of an HGR object."""

        def __init__(self) -> None:
            self.backend: Optional[Backend] = None
            self.last_result: Optional[HGR.Result] = None
            self.num_calls: int = 0

    backend: str = field(default='numpy')
    """The type of backend used for HGR computation."""

    _state: _State = field(init=False, default_factory=_State)
    """The internal mutable state of the HGR object."""

    def __post_init__(self) -> None:
        if self.backend == 'numpy':
            self._state.backend = NumpyBackend()
        elif self.backend == 'torch':
            self._state.backend = TorchBackend()
        else:
            raise AssertionError(f"Unknown backend '{self.backend}'")

    @property
    def last_result(self) -> Optional[Result]:
        """The `Result` instance returned from the last HGR call, or None if no call was performed."""
        return self._state.last_result

    @property
    def num_calls(self) -> int:
        """The number of times that this HGR instance was called."""
        return self._state.num_calls

    def f(self, a) -> Any:
        """Returns the mapped vector f(a) using the kernel function f computed in the last execution.

        :param a:
            The vector to be projected.

        :return:
            The resulting projection.
        """
        assert self._state.last_result is not None, "HGR has not been computed yet, so no kernel can be used."
        return self._f(a=a)

    def g(self, b) -> Any:
        """Returns the mapped vector g(b) using the kernel function g computed in the last execution.

        :param b:
            The vector to be projected.

        :return:
            The resulting projection.
        """
        assert self._state.last_result is not None, "HGR has not been computed yet, so no kernel can be used."
        return self._g(b=b)

    def correlation(self, a, b) -> Any:
        """Computes the HGR correlation.

        :param a:
            The first vector.

        :param b:
            The second vector.

        :result:
            A scalar value representing the computed correlation, optionally with gradient information attached."""
        return self(a=a, b=b).correlation

    def __call__(self, a, b) -> Result:
        """Computes the HGR correlation.

        :param a:
            The first vector.

        :param b:
            The second vector.

        :result:
            A `Result` instance containing the computed correlation together with additional information.
        """
        bk = self._state.backend
        assert bk.ndim(a) == bk.ndim(b) == 1, f"Expected vectors with one dimension, got {bk.ndim(a)} and {bk.ndim(b)}"
        assert bk.len(a) == bk.len(b), f"Input vectors must have the same dimension, got {bk.len(a)} != {bk.len(b)}"
        res = self._compute(a=a, b=b)
        self._state.last_result = res
        self._state.num_calls += 1
        return res

    @abstractmethod
    def _compute(self, a: np.ndarray, b: np.ndarray) -> Result:
        pass

    @abstractmethod
    def _f(self, a) -> Any:
        pass

    @abstractmethod
    def _g(self, b) -> Any:
        pass
