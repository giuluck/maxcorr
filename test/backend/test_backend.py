import unittest
from typing import Callable

import numpy as np

from cfair.backend import Backend


class TestBackend(unittest.TestCase):

    def _test_errors(self, backend: Backend, vector_fn: Callable[..., np.ndarray]) -> None:
        """Preliminary test to check that the backend operations do not raise any error."""
        scalar = 5
        vector = vector_fn(10)
        matrix = vector_fn(10, 3)
        operations = {
            'comply': [vector],
            'cast': [vector],
            'numpy': [vector],
            'zeros': [scalar],
            'ones': [scalar],
            'dtype': [vector],
            'shape': [vector],
            'ndim': [vector],
            'len': [vector],
            'stack': [[vector] * 3],
            'abs': [vector],
            'square': [vector],
            'sqrt': [vector],
            'matmul': [vector, vector],
            'maximum': [vector, vector],
            'mean': [vector],
            'var': [vector],
            'std': [vector],
            'standardize': [vector],
            'lstsq': [matrix, vector]
        }
        for operation, inputs in operations.items():
            try:
                function = getattr(backend, operation)
                function(*inputs)
            except Exception as exception:
                self.fail(f"Operation '{operation}' failed on {backend}:\n{exception}")
