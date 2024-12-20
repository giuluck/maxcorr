from typing import Type, List

from cfair.indicators import Indicator, NeuralHGR, NeuralGeDI, NeuralNLC
from test.indicators.test_indicator import TestIndicator


class TestNeuralHGR(TestIndicator):
    def indicators(self, backend: str) -> List[Indicator]:
        return [
            NeuralHGR(backend=backend),
            NeuralHGR(backend=backend, f_units=None),
            NeuralHGR(backend=backend, g_units=None)
        ]

    @property
    def result_type(self) -> Type:
        return NeuralHGR.Result


class TestNeuralGeDI(TestIndicator):
    def indicators(self, backend: str) -> List[Indicator]:
        return [
            NeuralGeDI(backend=backend),
            NeuralGeDI(backend=backend, f_units=None),
            NeuralGeDI(backend=backend, g_units=None)
        ]

    @property
    def result_type(self) -> Type:
        return NeuralGeDI.Result


class TestNeuralNLC(TestIndicator):
    def indicators(self, backend: str) -> List[Indicator]:
        return [
            NeuralNLC(backend=backend),
            NeuralNLC(backend=backend, f_units=None),
            NeuralNLC(backend=backend, g_units=None)
        ]

    @property
    def result_type(self) -> Type:
        return NeuralNLC.Result
