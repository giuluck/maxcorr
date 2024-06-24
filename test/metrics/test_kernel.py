from typing import Type, List

from cfair.metrics import Metric, DoubleKernelHGR, SingleKernelHGR
from cfair.metrics.kernel.gedi import SymmetricGeDI, GeneralizedDisparateImpact
from test.metrics.test_metric import TestMetric


class TestDoubleKernelHGR(TestMetric):
    def metrics(self, backend: str) -> List[Metric]:
        return [
            DoubleKernelHGR(backend=backend),
            DoubleKernelHGR(backend=backend, use_lstsq=False),
            DoubleKernelHGR(backend=backend, kernel_a=1, kernel_b=1)
        ]

    @property
    def result_type(self) -> Type:
        return DoubleKernelHGR.Result


class TestSingleKernelHGR(TestMetric):
    def metrics(self, backend: str) -> List[Metric]:
        return [
            SingleKernelHGR(backend=backend),
            SingleKernelHGR(backend=backend, use_lstsq=False),
            SingleKernelHGR(backend=backend, kernel=1)
        ]

    @property
    def result_type(self) -> Type:
        return SingleKernelHGR.Result


class TestSymmetricGeDI(TestMetric):
    def metrics(self, backend: str) -> List[Metric]:
        return [
            SymmetricGeDI(backend=backend),
            SymmetricGeDI(backend=backend, use_lstsq=False),
            SymmetricGeDI(backend=backend, kernel=1)
        ]

    @property
    def result_type(self) -> Type:
        return SymmetricGeDI.Result


class TestGeneralizedDisparateImpact(TestMetric):
    def metrics(self, backend: str) -> List[Metric]:
        return [
            GeneralizedDisparateImpact(backend=backend),
            GeneralizedDisparateImpact(backend=backend, use_lstsq=False),
            GeneralizedDisparateImpact(backend=backend, kernel=1)
        ]

    @property
    def result_type(self) -> Type:
        return GeneralizedDisparateImpact.Result
