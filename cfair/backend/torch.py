from typing import Any, Type

import numpy as np

from cfair.backend import Backend


class TorchBackend(Backend):
    def __init__(self) -> None:
        try:
            import torch
            super(TorchBackend, self).__init__(backend=torch)
        except ModuleNotFoundError:
            raise ModuleNotFoundError("TorchBackend requires 'torch', please install it via 'pip install torch'")

    @property
    def type(self) -> Type:
        return self._backend.Tensor

    def cast(self, v, dtype=None) -> Any:
        return self._backend.tensor(v, dtype=dtype)

    def numpy(self, v, dtype=None) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return v.detach().cpu().numpy()

    def stack(self, v: list) -> Any:
        return self._backend.stack(v, dim=1)

    def matmul(self, v, w) -> Any:
        return self._backend.matmul(v, w)

    def mean(self, v) -> Any:
        return self._backend.mean(v)

    def var(self, v) -> Any:
        return self._backend.var(v, unbiased=False)

    def lstsq(self, a, b) -> Any:
        # the 'gelsd' driver allows to have both more precise and more reproducible results
        return self._backend.linalg.lstsq(a, b, driver='gelsd')[0]
