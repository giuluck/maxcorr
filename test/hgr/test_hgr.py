import unittest
from abc import abstractmethod
from typing import final, Type, List

import numpy as np
import pytest

from cfair.backend import Backend, NumpyBackend, TensorflowBackend, TorchBackend
from cfair.hgr.hgr import HGR


class TestHGR(unittest.TestCase):
    RUNS: int = 5

    LENGTH: int = 10

    BACKENDS: List[Backend] = [TensorflowBackend(), TorchBackend(), NumpyBackend()]

    @property
    @abstractmethod
    def hgr_type(self) -> Type[HGR]:
        pytest.skip(reason="Abstract Test Class")

    @final
    @property
    def result_type(self) -> Type:
        return self.hgr_type.Result

    @final
    def hgr(self, backend: Backend) -> HGR:
        return self.hgr_type(backend=backend)

    @final
    def vectors(self, *seeds: int, backend: Backend) -> list:
        return [backend.cast(v=np.random.default_rng(seed=s).normal(size=self.LENGTH), dtype=float) for s in seeds]

    @final
    def test_correlation(self) -> None:
        # perform a simple sanity check on the stored result
        for backend in self.BACKENDS:
            vec1, vec2 = self.vectors(0, 1, backend=backend)
            hgr = self.hgr(backend=backend.name)
            self.assertEqual(
                hgr.correlation(a=vec1, b=vec2),
                hgr.last_result.correlation,
                msg=f"Inconsistent correlation between HGR method and result instance on {backend}"
            )

    @final
    def test_result(self) -> None:
        for backend in self.BACKENDS:
            vec1, vec2 = self.vectors(0, 1, backend=backend)
            hgr = self.hgr(backend=backend.name)
            result = hgr(a=vec1, b=vec2)
            self.assertIsInstance(result, self.result_type, msg=f"Wrong result class type from HGR call on {backend}")
            self.assertEqual(result, hgr.last_result, msg=f"Wrong result stored or yielded from HGR call on {backend}")
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
                result.correlation,
                (float, backend.type),
                msg=f"Wrong correlation type from HGR result on {backend}"
            )
            self.assertEqual(result.num_call, 1, msg=f"Wrong number of calls stored in result instance on {backend}")
            self.assertEqual(hgr, result.hgr, msg=f"Wrong HGR instance stored in result instance on {backend}")

    @final
    def test_state(self) -> None:
        for backend in self.BACKENDS:
            hgr = self.hgr(backend=backend.name)
            self.assertIsNone(hgr.last_result, msg=f"Wrong initial last result for HGR on {backend}")
            self.assertEqual(hgr.num_calls, 0, msg=f"Wrong initial number of calls stored in HGR on {backend}")
            results = []
            for i in range(self.RUNS):
                vec1, vec2 = self.vectors(i, i + self.RUNS, backend=backend)
                results.append(hgr(a=vec1, b=vec2))
                self.assertEqual(hgr.last_result, results[i], msg=f"Wrong last result for HGR on {backend}")
                self.assertEqual(hgr.num_calls, i + 1, msg=f"Wrong number of calls stored in HGR on {backend}")
            for i, result in enumerate(results):
                self.assertEqual(
                    result.num_call,
                    i + 1,
                    msg=f"Inconsistent number of call stored in HGR result on {backend}"
                )
