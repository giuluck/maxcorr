from typing import Type, List

from cfair import DoubleKernelGeDI, SingleKernelGeDI, DoubleKernelNLC, SingleKernelNLC
from cfair.indicators import Indicator, DoubleKernelHGR, SingleKernelHGR
from test.indicators.test_indicator import TestIndicator


class TestDoubleKernelHGR(TestIndicator):
    def indicators(self, backend: str) -> List[Indicator]:
        return [
            DoubleKernelHGR(backend=backend, kernel_a=3, kernel_b=3, use_lstsq=False),
            DoubleKernelHGR(backend=backend, kernel_a=3, kernel_b=1, use_lstsq=False),
            DoubleKernelHGR(backend=backend, kernel_a=3, kernel_b=1, use_lstsq=True),
            DoubleKernelHGR(backend=backend, kernel_a=1, kernel_b=3, use_lstsq=False),
            DoubleKernelHGR(backend=backend, kernel_a=1, kernel_b=3, use_lstsq=True),
            DoubleKernelHGR(backend=backend, kernel_a=1, kernel_b=1),
        ]

    @property
    def result_type(self) -> Type:
        return DoubleKernelHGR.Result


class TestSingleKernelHGR(TestIndicator):
    def indicators(self, backend: str) -> List[Indicator]:
        return [
            SingleKernelHGR(backend=backend, kernel=3, use_lstsq=False),
            SingleKernelHGR(backend=backend, kernel=3, use_lstsq=True),
            SingleKernelHGR(backend=backend, kernel=1)
        ]

    @property
    def result_type(self) -> Type:
        return SingleKernelHGR.Result


class TestDoubleKernelGeDI(TestIndicator):
    def indicators(self, backend: str) -> List[Indicator]:
        return [
            DoubleKernelGeDI(backend=backend, kernel_a=3, kernel_b=3, use_lstsq=False),
            DoubleKernelGeDI(backend=backend, kernel_a=3, kernel_b=1, use_lstsq=False),
            DoubleKernelGeDI(backend=backend, kernel_a=3, kernel_b=1, use_lstsq=True),
            DoubleKernelGeDI(backend=backend, kernel_a=1, kernel_b=3, use_lstsq=False),
            DoubleKernelGeDI(backend=backend, kernel_a=1, kernel_b=3, use_lstsq=True),
            DoubleKernelGeDI(backend=backend, kernel_a=1, kernel_b=1),
        ]

    @property
    def result_type(self) -> Type:
        return DoubleKernelGeDI.Result


class TestSingleKernelGeDI(TestIndicator):
    def indicators(self, backend: str) -> List[Indicator]:
        return [
            SingleKernelGeDI(backend=backend, kernel=3, use_lstsq=False),
            SingleKernelGeDI(backend=backend, kernel=3, use_lstsq=True),
            SingleKernelGeDI(backend=backend, kernel=1)
        ]

    @property
    def result_type(self) -> Type:
        return SingleKernelGeDI.Result


class TestDoubleKernelNLC(TestIndicator):
    def indicators(self, backend: str) -> List[Indicator]:
        return [
            DoubleKernelNLC(backend=backend, kernel_a=3, kernel_b=3, use_lstsq=False),
            DoubleKernelNLC(backend=backend, kernel_a=3, kernel_b=1, use_lstsq=False),
            DoubleKernelNLC(backend=backend, kernel_a=3, kernel_b=1, use_lstsq=True),
            DoubleKernelNLC(backend=backend, kernel_a=1, kernel_b=3, use_lstsq=False),
            DoubleKernelNLC(backend=backend, kernel_a=1, kernel_b=3, use_lstsq=True),
            DoubleKernelNLC(backend=backend, kernel_a=1, kernel_b=1),
        ]

    @property
    def result_type(self) -> Type:
        return DoubleKernelNLC.Result


class TestSingleKernelNLC(TestIndicator):
    def indicators(self, backend: str) -> List[Indicator]:
        return [
            SingleKernelNLC(backend=backend, kernel=3, use_lstsq=False),
            SingleKernelNLC(backend=backend, kernel=3, use_lstsq=True),
            SingleKernelNLC(backend=backend, kernel=1)
        ]

    @property
    def result_type(self) -> Type:
        return SingleKernelNLC.Result
