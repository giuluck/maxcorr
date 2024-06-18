from typing import Type

from cfair.backend import Backend
from cfair.hgr import DoubleKernelHGR, SingleKernelHGR, HGR
from test.hgr.test_hgr import TestHGR


class TestDoubleKernelHGR(TestHGR):

    def hgr(self, backend: Backend) -> HGR:
        return DoubleKernelHGR(backend=backend)

    @property
    def result_type(self) -> Type:
        return DoubleKernelHGR.Result


class TestSingleKernelHGR(TestHGR):

    def hgr(self, backend: Backend) -> HGR:
        return SingleKernelHGR(backend=backend)

    @property
    def result_type(self) -> Type:
        return SingleKernelHGR.Result
