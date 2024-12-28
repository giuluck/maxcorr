from typing import Type, List, Tuple

from maxcorr import BackendType, SemanticsType
from maxcorr.indicators import Indicator, DensityIndicator
from test.indicators.test_indicator import TestIndicator


class TestDensityIndicator(TestIndicator):
    def indicators(self, backend: BackendType, semantics: SemanticsType, dim: Tuple[int, int]) -> List[Indicator]:
        return [
            DensityIndicator(backend=backend, semantics=semantics, chi_square=False),
            DensityIndicator(backend=backend, semantics=semantics, chi_square=True)
        ] if dim == (1, 1) else []

    @property
    def result_type(self) -> Type:
        return DensityIndicator.Result
