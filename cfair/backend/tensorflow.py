import importlib.util
from typing import Any, Type

import numpy as np

from cfair.backend import Backend


class TensorflowBackend(Backend):
    def __init__(self):
        if importlib.util.find_spec('tensorflow') is None:
            raise ModuleNotFoundError(
                "TensorflowBackend requires 'tensorflow', please install it via 'pip install tensorflow'"
            )
        import tensorflow
        super(TensorflowBackend, self).__init__(backend=tensorflow)

    @property
    def type(self) -> Type:
        return self._backend.Tensor

    def cast(self, v, dtype=None) -> Any:
        return self._backend.constant(v, dtype=dtype)

    def numpy(self, v, dtype=None) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return v.numpy()

    def stack(self, v: list) -> Any:
        return self._backend.stack(v, axis=1)

    def matmul(self, v, w) -> Any:
        v = self.reshape(v, shape=(1, -1)) if self.ndim(v) == 1 else v
        w = self.reshape(w, shape=(-1, 1)) if self.ndim(w) == 1 else w
        s = self._backend.linalg.matmul(v, w)
        return self.reshape(s, shape=-1)

    def mean(self, v) -> Any:
        return self._backend.math.reduce_mean(v)

    def var(self, v) -> Any:
        return self._backend.math.reduce_variance(v)

    def lstsq(self, a, b) -> Any:
        # use fast=False to obtain more robust results
        b = self.reshape(b, shape=(-1, 1))
        w = self._backend.linalg.lstsq(a, b, fast=False)
        return self.reshape(w, shape=-1)
