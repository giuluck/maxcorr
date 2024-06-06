from cfair.hgr import DoubleKernelHGR, SingleKernelHGR
from test.hgr.test_hgr import TestHGR


class TestDoubleKernelHGR(TestHGR):
    def test_errors(self) -> None:
        return self._test_errors(
            hgr_fn=lambda backend: DoubleKernelHGR(backend=backend.name)
        )


class TestSingleKernelHGR(TestHGR):
    def test_errors(self) -> None:
        return self._test_errors(
            hgr_fn=lambda backend: SingleKernelHGR(backend=backend.name)
        )
