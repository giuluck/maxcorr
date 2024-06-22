from typing import Type

from cfair.metrics import Metric, AdversarialHGR
from test.metrics.test_metric import TestMetric


class TestAdversarialHGR(TestMetric):
    @property
    def metric_type(self) -> Type[Metric]:
        return AdversarialHGR
