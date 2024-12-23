from typing import Type, List

from cfair import BackendType, SemanticsType
from cfair.indicators import Indicator, DensityIndicator
from test.indicators.test_indicator import TestIndicator


class TestDensityHGR(TestIndicator):
    def indicators(self, backend: BackendType, semantics: SemanticsType) -> List[Indicator]:
        return [
            DensityIndicator(backend=backend, semantics=semantics, chi_square=False),
            DensityIndicator(backend=backend, semantics=semantics, chi_square=True)
        ]

    @property
    def result_type(self) -> Type:
        return DensityIndicator.Result
