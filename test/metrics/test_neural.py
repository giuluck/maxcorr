from typing import Type

from cfair.metrics import Metric, NeuralHGR
from test.metrics.test_metric import TestMetric


class TestNeuralHGR(TestMetric):
    @property
    def metric_type(self) -> Type[Metric]:
        return NeuralHGR
