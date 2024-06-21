from typing import Any, Type, Optional

import numpy as np

from cfair.backend import Backend


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
        # TODO: quick fix to handle cast of torch tensors with gradients
        try:
            return self._backend.array(v, dtype=dtype)
        except RuntimeError:
            return v.detach().cpu().numpy()

    def numpy(self, v, dtype=None) -> np.ndarray:
        return v

    def stack(self, v: list) -> Any:
        return self._backend.stack(v, axis=1)

    def matmul(self, v, w) -> Any:
        return self._backend.matmul(v, w)

    def mean(self, v) -> Any:
        return self._backend.mean(v)

    def var(self, v) -> Any:
        return self._backend.var(v, ddof=0)

    def lstsq(self, a, b) -> Any:
        return self._backend.linalg.lstsq(a, b, rcond=None)[0]
