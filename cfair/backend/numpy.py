from typing import Any

import numpy as np

from cfair.backend import Backend


class NumpyBackend(Backend):
    def __init__(self) -> None:
        super(NumpyBackend, self).__init__(backend=np)

    def comply(self, v) -> bool:
        return isinstance(v, np.ndarray)

    def cast(self, v, dtype=None) -> Any:
        return np.array(v, dtype=dtype)

    def numpy(self, v, dtype=None) -> np.ndarray:
        return v

    def stack(self, v: list) -> Any:
        return self._backend.stack(v, axis=1)

    def var(self, v) -> Any:
        return self._backend.var(v, ddof=0)

    def lstsq(self, a, b) -> Any:
        return self._backend.linalg.lstsq(a, b, rcond=None)[0]
