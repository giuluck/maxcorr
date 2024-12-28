from typing import Type, List, Tuple

from maxcorr import Indicator, DoubleKernelIndicator, BackendType, SemanticsType, SingleKernelIndicator
from test.indicators.test_indicator import TestIndicator


class TestDoubleKernelIndicator(TestIndicator):

    def indicators(self, backend: BackendType, semantics: SemanticsType, dim: Tuple[int, int]) -> List[Indicator]:
        return [
            DoubleKernelIndicator(backend=backend, semantics=semantics, kernel_a=3, kernel_b=3, use_lstsq=False),
            DoubleKernelIndicator(backend=backend, semantics=semantics, kernel_a=3, kernel_b=1, use_lstsq=False),
            DoubleKernelIndicator(backend=backend, semantics=semantics, kernel_a=3, kernel_b=1, use_lstsq=True),
            DoubleKernelIndicator(backend=backend, semantics=semantics, kernel_a=1, kernel_b=3, use_lstsq=False),
            DoubleKernelIndicator(backend=backend, semantics=semantics, kernel_a=1, kernel_b=3, use_lstsq=True),
            DoubleKernelIndicator(backend=backend, semantics=semantics, kernel_a=1, kernel_b=1),
        ]

    @property
    def result_type(self) -> Type:
        return DoubleKernelIndicator.Result


class TestSingleKernelIndicator(TestIndicator):
    def indicators(self, backend: BackendType, semantics: SemanticsType, dim: Tuple[int, int]) -> List[Indicator]:
        return [
            SingleKernelIndicator(backend=backend, semantics=semantics, kernel=3, use_lstsq=False),
            SingleKernelIndicator(backend=backend, semantics=semantics, kernel=3, use_lstsq=True),
            SingleKernelIndicator(backend=backend, semantics=semantics, kernel=1)
        ]

    @property
    def result_type(self) -> Type:
        return SingleKernelIndicator.Result
