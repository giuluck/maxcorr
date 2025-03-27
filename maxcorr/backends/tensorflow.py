import importlib.util
from typing import Any, Type, Union, Iterable

import numpy as np

from maxcorr.backends import Backend


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

    def floating(self, v) -> bool:
        return v.dtype.is_floating

    def cast(self, v, dtype=None) -> Any:
        # detach torch tensor if passed as input
        if importlib.util.find_spec('torch') is not None:
            import torch
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
        # build a dictionary to avoid passing dtype=None to tensorflow primitives
        kwargs = dict() if dtype is None else dict(dtype=dtype)
        # if the vector is already a tf tensor, simply change the dtype to avoid warnings
        return self._backend.cast(v, **kwargs) if self.comply(v) else self._backend.constant(v, **kwargs)

    def item(self, v) -> float:
        return float(v.numpy().item())

    def numpy(self, v, dtype=None) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return v.numpy()

    def stack(self, v: list, axis: Union[int, Iterable[int]] = 0) -> Any:
        return self._backend.stack(v, axis=axis)

    def matmul(self, v, w) -> Any:
        v = self.reshape(v, shape=(1, -1)) if self.ndim(v) == 1 else v
        w = self.reshape(w, shape=(-1, 1)) if self.ndim(w) == 1 else w
        return self._backend.squeeze(self._backend.linalg.matmul(v, w))

    def mean(self, v, axis: Union[None, int, Iterable[int]] = None) -> Any:
        return self._backend.math.reduce_mean(v, axis=axis)

    def sum(self, v, axis: Union[None, int, Iterable[int]] = None) -> Any:
        return self._backend.math.reduce_sum(v, axis=axis)

    def cov(self, v, w) -> Any:
        inp = self.stack([v, w], axis=1)
        inp = inp - self.mean(inp, axis=0)
        return self.matmul(self._backend.transpose(inp), inp) / self.len(v)

    def var(self, v, axis: Union[None, int, Iterable[int]] = None) -> Any:
        return self._backend.math.reduce_variance(v, axis=axis)

    # noinspection PyPep8Naming
    def lstsq(self, A, b) -> Any:
        # use fast=False to obtain more robust results
        b = self.reshape(b, shape=(-1, 1))
        w = self._backend.linalg.lstsq(A, b, fast=False)
        return self.reshape(w, shape=-1)
