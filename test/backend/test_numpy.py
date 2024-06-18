from typing import Any, Type

import numpy as np

from cfair.backend import Backend, NumpyBackend
from test.backend.test_backend import TestBackend


class TestNumpyBackend(TestBackend):

    @property
    def backend(self) -> Backend:
        return NumpyBackend()

    @property
    def type(self) -> Type:
        return np.ndarray

    def cast(self, v: list) -> Any:
        return np.array(v)
