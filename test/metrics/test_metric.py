import unittest
from abc import abstractmethod
from typing import final, Type, List

import numpy as np
import pytest

from cfair.backends import Backend, NumpyBackend, TensorflowBackend, TorchBackend
from cfair.metrics.metric import Metric


class TestMetric(unittest.TestCase):
    RUNS: int = 5

    LENGTH: int = 10

    BACKENDS: List[Backend] = [TensorflowBackend(), TorchBackend(), NumpyBackend()]

    @property
    @abstractmethod
    def metric_type(self) -> Type[Metric]:
        pytest.skip(reason="Abstract Test Class")

    @final
    @property
    def result_type(self) -> Type:
        return self.metric_type.Result

    @final
    def metric(self, backend: Backend) -> Metric:
        return self.metric_type(backend=backend)

    @final
    def vectors(self, *seeds: int, backend: Backend) -> list:
        return [backend.cast(v=np.random.default_rng(seed=s).normal(size=self.LENGTH), dtype=float) for s in seeds]

    @final
    def test_value(self) -> None:
        # perform a simple sanity check on the stored result
        for backend in self.BACKENDS:
            vec1, vec2 = self.vectors(0, 1, backend=backend)
            metric = self.metric(backend=backend.name)
            self.assertEqual(
                metric.value(a=vec1, b=vec2),
                metric.last_result.value,
                msg=f"Inconsistent return between 'value' method and result instance on {backend}"
            )

    @final
    def test_result(self) -> None:
        for backend in self.BACKENDS:
            vec1, vec2 = self.vectors(0, 1, backend=backend)
            metric = self.metric(backend=backend.name)
            result = metric(a=vec1, b=vec2)
            self.assertIsInstance(result, self.result_type, msg=f"Wrong result class type from 'call' on {backend}")
            self.assertEqual(result, metric.last_result, msg=f"Wrong result stored or yielded from 'call' on {backend}")
            self.assertEqual(
                backend.numpy(vec1).tolist(),
                backend.numpy(result.a).tolist(),
                msg=f"Wrong 'a' vector stored in result instance on {backend}"
            )
            self.assertEqual(
                backend.numpy(vec2).tolist(),
                backend.numpy(result.b).tolist(),
                msg=f"Wrong 'b' vector stored in result instance on {backend}"
            )
            # include "float" in types since numpy arrays return floats for aggregated operations
            self.assertIsInstance(
                result.value,
                (float, backend.type),
                msg=f"Wrong value type from result instance on {backend}"
            )
            self.assertEqual(result.num_call, 1, msg=f"Wrong number of calls stored in result instance on {backend}")
            self.assertEqual(metric, result.metric, msg=f"Wrong metric instance stored in result instance on {backend}")

    @final
    def test_state(self) -> None:
        for backend in self.BACKENDS:
            metric = self.metric(backend=backend.name)
            self.assertIsNone(metric.last_result, msg=f"Wrong initial last result on {backend}")
            self.assertEqual(metric.num_calls, 0, msg=f"Wrong initial number of calls stored on {backend}")
            results = []
            for i in range(self.RUNS):
                vec1, vec2 = self.vectors(i, i + self.RUNS, backend=backend)
                results.append(metric(a=vec1, b=vec2))
                self.assertEqual(metric.last_result, results[i], msg=f"Wrong last result on {backend}")
                self.assertEqual(metric.num_calls, i + 1, msg=f"Wrong number of calls stored on {backend}")
            for i, result in enumerate(results):
                self.assertEqual(
                    result.num_call,
                    i + 1,
                    msg=f"Inconsistent number of call stored in returned result on {backend}"
                )
