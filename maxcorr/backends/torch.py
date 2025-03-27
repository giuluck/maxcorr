import importlib.util
from typing import Any, Type, Union, Iterable

import numpy as np

from maxcorr.backends import Backend


class TorchBackend(Backend):
    def __init__(self):
        if importlib.util.find_spec('torch') is None:
            raise ModuleNotFoundError("TorchBackend requires 'torch', please install it via 'pip install torch'")
        import torch
        super(TorchBackend, self).__init__(backend=torch)

    @property
    def type(self) -> Type:
        return self._backend.Tensor

    def floating(self, v) -> bool:
        return self._backend.is_floating_point(v)

    def cast(self, v, dtype=None) -> Any:
        # torch uses 'torch.float64' as default type for 'float', use 'torch.float32' instead for compatibility
        dtype = self._backend.float32 if dtype is float else dtype
        # if the vector is already a torch tensor, simply change the dtype to avoid warnings
        return v.to(dtype=dtype) if self.comply(v) else self._backend.tensor(v, dtype=dtype)

    def item(self, v) -> float:
        return float(v.item())

    def numpy(self, v, dtype=None) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return v.detach().cpu().numpy()

    def stack(self, v: list, axis: Union[int, Iterable[int]] = 0) -> Any:
        kwargs = dict() if axis is None else dict(dim=axis)
        return self._backend.stack(v, **kwargs)

    def matmul(self, v, w) -> Any:
        return self._backend.matmul(v, w)

    def transpose(self, v: Any) -> Any:
        return self.reshape(v, shape=(self.len(v), -1)).T

    def mean(self, v, axis: Union[None, int, Iterable[int]] = None) -> Any:
        kwargs = dict() if axis is None else dict(dim=axis)
        return self._backend.mean(v, **kwargs)

    def sum(self, v, axis: Union[None, int, Iterable[int]] = None) -> Any:
        kwargs = dict() if axis is None else dict(dim=axis)
        return self._backend.sum(v, **kwargs)

    def cov(self, v, w) -> Any:
        inp = self.stack([v, w])
        return self._backend.cov(inp, correction=0)

    def var(self, v, axis: Union[None, int, Iterable[int]] = None) -> Any:
        kwargs = dict() if axis is None else dict(dim=axis)
        return self._backend.var(v, unbiased=False, **kwargs)

    # noinspection PyPep8Naming
    def lstsq(self, A, b) -> Any:
        # the 'gelsd' driver allows to have both more precise and more reproducible results
        return self._backend.linalg.lstsq(A, b, driver='gelsd')[0]
