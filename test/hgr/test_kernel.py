from typing import Type

from cfair.hgr import DoubleKernelHGR, SingleKernelHGR, HGR
from test.hgr.test_hgr import TestHGR


class TestDoubleKernelHGR(TestHGR):

    @property
    def hgr_type(self) -> Type[HGR]:
        return DoubleKernelHGR


class TestSingleKernelHGR(TestHGR):

    @property
    def hgr_type(self) -> Type[HGR]:
        return SingleKernelHGR
