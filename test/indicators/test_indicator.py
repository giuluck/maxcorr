import unittest
from abc import abstractmethod
from typing import Type, List, Dict

import numpy as np
import pytest

from cfair import BackendType, SemanticsType
from cfair.backends import Backend, NumpyBackend, TensorflowBackend, TorchBackend
from cfair.indicators.indicator import Indicator


class TestIndicator(unittest.TestCase):
    RUNS: int = 5

    LENGTH: int = 10

    BACKENDS: Dict[BackendType, Backend] = {
        'numpy': NumpyBackend(),
        'torch': TorchBackend(),
        'tensorflow': TensorflowBackend()
    }

    SEMANTICS: List[SemanticsType] = ['hgr', 'gedi', 'nlc']

    # noinspection PyTypeChecker
    @abstractmethod
    def indicators(self, backend: BackendType, semantics: SemanticsType) -> List[Indicator]:
        pytest.skip(reason="Abstract Test Class")

    # noinspection PyTypeChecker
    @property
    @abstractmethod
    def result_type(self) -> Type:
        pytest.skip(reason="Abstract Test Class")

    def vectors(self, *seeds: int, backend: Backend) -> list:
        return [backend.cast(v=np.random.default_rng(seed=s).normal(size=self.LENGTH), dtype=float) for s in seeds]

    def test_value(self) -> None:
        # perform a simple sanity check on the stored result
        for bk, backend in self.BACKENDS.items():
            for sm in self.SEMANTICS:
                vec1, vec2 = self.vectors(0, 1, backend=backend)
                for mt in self.indicators(backend=bk, semantics=sm):
                    self.assertEqual(
                        mt.compute(a=vec1, b=vec2),
                        mt.last_result.value,
                        msg=f"Inconsistent return between 'value' method and result instance on {bk}"
                    )

    def test_result(self) -> None:
        for bk, backend in self.BACKENDS.items():
            for sm in self.SEMANTICS:
                vec1, vec2 = self.vectors(0, 1, backend=backend)
                for mt in self.indicators(backend=bk, semantics=sm):
                    result = mt(a=vec1, b=vec2)
                    self.assertIsInstance(result, self.result_type, msg=f"Wrong result class type from 'call' on {bk}")
                    self.assertEqual(result, mt.last_result, msg=f"Wrong result stored or yielded from 'call' on {bk}")
                    self.assertEqual(
                        backend.numpy(vec1).tolist(),
                        backend.numpy(result.a).tolist(),
                        msg=f"Wrong 'a' vector stored in result instance on {bk}"
                    )
                    self.assertEqual(
                        backend.numpy(vec2).tolist(),
                        backend.numpy(result.b).tolist(),
                        msg=f"Wrong 'b' vector stored in result instance on {bk}"
                    )
                    # include "float" in types since numpy arrays return floats for aggregated operations
                    self.assertIsInstance(
                        result.value,
                        (float, backend.type),
                        msg=f"Wrong value type from result instance on {bk}"
                    )
                    self.assertEqual(result.num_call, 1, msg=f"Wrong number of calls stored in result instance on {bk}")
                    self.assertEqual(mt, result.indicator, msg=f"Wrong indicator stored in result instance on {bk}")

    def test_state(self) -> None:
        for bk, backend in self.BACKENDS.items():
            for sm in self.SEMANTICS:
                for mt in self.indicators(backend=bk, semantics=sm):
                    self.assertIsNone(mt.last_result, msg=f"Wrong initial last result on {bk}")
                    self.assertEqual(mt.num_calls, 0, msg=f"Wrong initial number of calls stored on {bk}")
                    results = []
                    for i in range(self.RUNS):
                        vec1, vec2 = self.vectors(i, i + self.RUNS, backend=backend)
                        results.append(mt(a=vec1, b=vec2))
                        self.assertEqual(mt.last_result, results[i], msg=f"Wrong last result on {bk}")
                        self.assertEqual(mt.num_calls, i + 1, msg=f"Wrong number of calls stored on {bk}")
                    for i, result in enumerate(results):
                        self.assertEqual(
                            result.num_call,
                            i + 1,
                            msg=f"Inconsistent number of call stored in returned result on {bk}"
                        )
