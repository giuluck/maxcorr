from typing import Any, Type

import numpy
import numpy as np

from cfair.backend import Backend


class NumpyBackend(Backend):

    def __init__(self) -> None:
        super(NumpyBackend, self).__init__(backend=np)

    @property
    def type(self) -> Type:
        return np.ndarray

    def cast(self, v, dtype=None) -> Any:
        return np.array(v, dtype=dtype)

    def numpy(self, v, dtype=None) -> np.ndarray:
        return v

    def stack(self, v: list) -> Any:
        return self._backend.stack(v, axis=1)

    def var(self, v) -> Any:
        return self._backend.var(v, ddof=0)

    def lstsq(self, a, b) -> Any:
        return self._backend.linalg.lstsq(a, b, rcond=None)[0]
