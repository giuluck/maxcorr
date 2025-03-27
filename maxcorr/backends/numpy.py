import importlib.util
from typing import Any, Type, Optional, Union, Iterable

import numpy as np

from maxcorr.backends import Backend


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

    def floating(self, v) -> bool:
        return np.issubdtype(v.dtype, np.floating)

    def cast(self, v, dtype=None) -> Any:
        # detach torch tensor if passed as input
        if importlib.util.find_spec('torch') is not None:
            import torch
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
        v = self._backend.array(v, dtype=dtype)
        return self.item(v) if v.ndim == 0 else v

    def item(self, v) -> float:
        return float(v.item())

    def numpy(self, v, dtype=None) -> np.ndarray:
        return v

    def stack(self, v: list, axis: Union[int, Iterable[int]] = 0) -> Any:
        return self._backend.stack(v, axis=axis)

    def matmul(self, v, w) -> Any:
        return self._backend.matmul(v, w)

    def mean(self, v, axis: Union[None, int, Iterable[int]] = None) -> Any:
        return self._backend.mean(v, axis=axis)

    def sum(self, v, axis: Union[None, int, Iterable[int]] = None) -> Any:
        return self._backend.sum(v, axis=axis)

    def cov(self, v, w) -> Any:
        return self._backend.cov(v, w, ddof=0)

    def var(self, v, axis: Union[None, int, Iterable[int]] = None) -> Any:
        return self._backend.var(v, axis=axis, ddof=0)

    # noinspection PyPep8Naming
    def lstsq(self, A, b) -> Any:
        return self._backend.linalg.lstsq(A, b, rcond=None)[0]
