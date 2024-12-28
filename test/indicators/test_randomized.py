from typing import Type, List, Tuple

import numpy as np

from maxcorr import BackendType, SemanticsType, RandomizedIndicator
from maxcorr.indicators import Indicator
from test.indicators.test_indicator import TestIndicator


class TestRandomizedIndicator(TestIndicator):
    def indicators(self, backend: BackendType, semantics: SemanticsType, dim: Tuple[int, int]) -> List[Indicator]:
        return [
            RandomizedIndicator(backend=backend, semantics=semantics, functions=np.sin),
            RandomizedIndicator(backend=backend, semantics=semantics, functions=np.cos)
        ] if dim == (1, 1) else []

    @property
    def result_type(self) -> Type:
        return RandomizedIndicator.Result
