from typing import Any, Type, Optional, Union, Iterable

import numpy as np

from cfair.backends import Backend


class NumpyBackend(Backend):
    _instance: Optional = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Backend, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        super(NumpyBackend, self).__init__(backend=np)

    @property
    def type(self) -> Type:
        return np.ndarray

    def cast(self, v, dtype=None) -> Any:
        return self._backend.array(v, dtype=dtype)

    def numpy(self, v, dtype=None) -> np.ndarray:
        return v

    def stack(self, v: list, axis: Union[None, int, Iterable[int]] = None) -> Any:
        return self._backend.stack(v, axis=axis)

    def matmul(self, v, w) -> Any:
        return self._backend.matmul(v, w)

    def mean(self, v, axis: Union[None, int, Iterable[int]] = None) -> Any:
        return self._backend.mean(v, axis=axis)

    def var(self, v, axis: Union[None, int, Iterable[int]] = None) -> Any:
        return self._backend.var(v, axis=axis, ddof=0)

    def lstsq(self, A, b) -> Any:
        return self._backend.linalg.lstsq(A, b, rcond=None)[0]
