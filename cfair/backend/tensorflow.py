from typing import Any, Type

import numpy as np

from cfair.backend import Backend


class TensorflowBackend(Backend):
    def __init__(self) -> None:
        try:
            import tensorflow
            super(TensorflowBackend, self).__init__(backend=tensorflow)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "TensorflowBackend requires 'tensorflow', please install it via 'pip install tensorflow'"
            )

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
        v = self._backend.reshape(v, (1, -1)) if self.ndim(v) == 1 else v
        w = self._backend.reshape(w, (-1, 1)) if self.ndim(w) == 1 else w
        s = self._backend.linalg.matmul(v, w)
        return self._backend.reshape(s, -1)

    def mean(self, v) -> Any:
        return self._backend.math.reduce_mean(v)

    def var(self, v) -> Any:
        return self._backend.math.reduce_variance(v)

    def lstsq(self, a, b) -> Any:
        # use fast=False to obtain more robust results
        b = self._backend.reshape(b, (-1, 1))
        w = self._backend.linalg.lstsq(a, b, fast=False)
        return self._backend.reshape(w, -1)
