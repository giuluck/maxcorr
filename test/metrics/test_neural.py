from typing import Type, List

from cfair.metrics import Metric, NeuralHGR
from test.metrics.test_metric import TestMetric


class TestNeuralHGR(TestMetric):
    def metrics(self, backend: str) -> List[Metric]:
        return [
            NeuralHGR(backend=backend),
            NeuralHGR(backend=backend, f_units=None),
            NeuralHGR(backend=backend, g_units=None)
        ]

    @property
    def result_type(self) -> Type:
        return NeuralHGR.Result
