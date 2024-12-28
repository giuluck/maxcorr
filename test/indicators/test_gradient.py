from typing import Type, List, Tuple

from maxcorr import NeuralIndicator, BackendType, SemanticsType, LatticeIndicator
from maxcorr.indicators import Indicator
from test.indicators.test_indicator import TestIndicator


class TestNeuralIndicator(TestIndicator):
    def indicators(self, backend: BackendType, semantics: SemanticsType, dim: Tuple[int, int]) -> List[Indicator]:
        out = [NeuralIndicator(backend=backend, semantics=semantics, num_features=dim)]
        if dim[0] == 1:
            out.append(NeuralIndicator(backend=backend, semantics=semantics, f_units=None, num_features=dim))
        if dim[1] == 1:
            out.append(NeuralIndicator(backend=backend, semantics=semantics, g_units=None, num_features=dim))
        if dim == (1, 1):
            out.append(NeuralIndicator(backend=backend, semantics=semantics, f_units=None, g_units=None))
        return out

    @property
    def result_type(self) -> Type:
        return NeuralIndicator.Result


class TestLatticeIndicator(TestIndicator):
    def indicators(self, backend: BackendType, semantics: SemanticsType, dim: Tuple[int, int]) -> List[Indicator]:
        df, dg = dim
        out = [LatticeIndicator(backend=backend, semantics=semantics, f_sizes=[10] * df, g_sizes=[10] * dg)]
        if df == 1:
            out.append(LatticeIndicator(backend=backend, semantics=semantics, f_sizes=None, g_sizes=[10] * dg))
        if dg == 1:
            out.append(LatticeIndicator(backend=backend, semantics=semantics, f_sizes=[10] * df, g_sizes=None))
        if dim == (1, 1):
            out.append(LatticeIndicator(backend=backend, semantics=semantics, f_sizes=None, g_sizes=None))
        return out

    @property
    def result_type(self) -> Type:
        return NeuralIndicator.Result
