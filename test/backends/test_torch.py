from typing import Any, Type

import torch

from maxcorr.backends import Backend, TorchBackend
from test.backends.test_backend import TestBackend


class TestTorchBackend(TestBackend):

    @property
    def backend(self) -> Backend:
        return TorchBackend()

    @property
    def type(self) -> Type:
        return torch.Tensor

    def cast(self, v: list) -> Any:
        return torch.tensor(v)
