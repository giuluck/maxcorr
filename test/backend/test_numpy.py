import numpy as np

from cfair.backend import NumpyBackend
from test.backend.test_backend import TestBackend


class TestNumpyBackend(TestBackend):
    def test_errors(self) -> None:
        return self._test_errors(
            backend=NumpyBackend(),
            vector_fn=lambda *shape: np.ones(shape)
        )
