from typing import Type, List, Tuple

from cfair import NeuralIndicator, BackendType, SemanticsType
from cfair.indicators import Indicator
from test.indicators.test_indicator import TestIndicator


class TestNeuralIndicator(TestIndicator):
    def indicators(self, backend: BackendType, semantics: SemanticsType, dim: Tuple[int, int]) -> List[Indicator]:
        out = [NeuralIndicator(backend=backend, semantics=semantics, num_features=dim)]
        if dim[0] == 1:
            out.append(NeuralIndicator(backend=backend, semantics=semantics, f_units=None, num_features=dim))
        if dim[1] == 1:
            out.append(NeuralIndicator(backend=backend, semantics=semantics, g_units=None, num_features=dim))
        return out

    @property
    def result_type(self) -> Type:
        return NeuralIndicator.Result
