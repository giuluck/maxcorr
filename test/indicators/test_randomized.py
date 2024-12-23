from typing import Type, List

import numpy as np

from cfair import BackendType, SemanticsType, RandomizedIndicator
from cfair.indicators import Indicator
from test.indicators.test_indicator import TestIndicator


class TestRandomizedIndicator(TestIndicator):
    def indicators(self, backend: BackendType, semantics: SemanticsType) -> List[Indicator]:
        return [
            RandomizedIndicator(backend=backend, semantics=semantics, functions=np.sin),
            RandomizedIndicator(backend=backend, semantics=semantics, functions=np.cos)
        ]

    @property
    def result_type(self) -> Type:
        return RandomizedIndicator.Result
