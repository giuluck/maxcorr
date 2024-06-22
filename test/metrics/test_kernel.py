from typing import Type

from cfair.metrics import DoubleKernelHGR, SingleKernelHGR, Metric
from test.metrics.test_metric import TestMetric


class TestDoubleKernelHGR(TestMetric):
    @property
    def metric_type(self) -> Type[Metric]:
        return DoubleKernelHGR


class TestSingleKernelHGR(TestMetric):
    @property
    def metric_type(self) -> Type[Metric]:
        return SingleKernelHGR
