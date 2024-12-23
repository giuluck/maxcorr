from typing import Type, List

from cfair import NeuralIndicator, BackendType, SemanticsType
from cfair.indicators import Indicator
from test.indicators.test_indicator import TestIndicator


class TestNeuralIndicator(TestIndicator):
    def indicators(self, backend: BackendType, semantics: SemanticsType) -> List[Indicator]:
        return [
            NeuralIndicator(backend=backend, semantics=semantics),
            NeuralIndicator(backend=backend, semantics=semantics, f_units=None),
            NeuralIndicator(backend=backend, semantics=semantics, g_units=None)
        ]

    @property
    def result_type(self) -> Type:
        return NeuralIndicator.Result
