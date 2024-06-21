from typing import Type

from cfair.hgr import HGR, AdversarialHGR
from test.hgr.test_hgr import TestHGR


class TestAdversarialHGR(TestHGR):
    @property
    def hgr_type(self) -> Type[HGR]:
        return AdversarialHGR
