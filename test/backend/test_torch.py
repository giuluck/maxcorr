import torch

from cfair.backend import TorchBackend
from test.backend.test_backend import TestBackend


class TestTorchBackend(TestBackend):
    def test_errors(self) -> None:
        return self._test_errors(
            backend=TorchBackend(),
            vector_fn=lambda *shape: torch.ones(shape)
        )
