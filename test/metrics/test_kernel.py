from typing import Type

from cfair.metrics import Metric, DoubleKernelHGR, SingleKernelHGR
from test.metrics.test_metric import TestMetric


class TestDoubleKernelHGR(TestMetric):
    @property
    def metric_type(self) -> Type[Metric]:
        return DoubleKernelHGR


class TestSingleKernelHGR(TestMetric):
    @property
    def metric_type(self) -> Type[Metric]:
        return SingleKernelHGR

# class TestDoubleKernelGeDI(TestMetric):
#     @property
#     def metric_type(self) -> Type[Metric]:
#         return DoubleKernelGeDI
#
#
# class TestSingleKernelGeDI(TestMetric):
#     @property
#     def metric_type(self) -> Type[Metric]:
#         return SingleKernelGeDI
#
#
# class TestGeneralizedDisparateImpact(TestMetric):
#     @property
#     def metric_type(self) -> Type[Metric]:
#         return GeneralizedDisparateImpact
