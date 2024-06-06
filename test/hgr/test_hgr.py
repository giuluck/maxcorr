import unittest
from typing import Callable

from cfair.backend import Backend
from cfair.backend import NumpyBackend, TorchBackend
from cfair.hgr.hgr import HGR


class TestHGR(unittest.TestCase):

    def _test_errors(self, hgr_fn: Callable[[Backend], HGR]) -> None:
        """Preliminary test to check that the operations on the HGR algorithm do not raise any error."""
        for backend in [NumpyBackend(), TorchBackend()]:
            hgr = hgr_fn(backend)
            vector = backend.ones(10)
            operations = {
                'correlation': [vector, vector],
                '__call__': [vector, vector],
                'f': [vector],
                'g': [vector],
                'last_result': None,
                'num_calls': None,
                'backend': None
            }
            for operation, inputs in operations.items():
                try:
                    function = getattr(hgr, operation)
                    if inputs is not None:
                        function(*inputs)
                except Exception as exception:
                    self.fail(f"Operation '{operation}' failed on {backend}:\n{exception}")
