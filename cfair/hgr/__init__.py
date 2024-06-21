from typing import Union

from cfair.backend import Backend
from cfair.hgr.hgr import HGR
from cfair.hgr.kernel import DoubleKernelHGR, SingleKernelHGR
from cfair.hgr.adversarial import AdversarialHGR


def hgr(backend: Union[str, Backend], algorithm: str, **kwargs) -> HGR:
    """Builds an HGR instance.

    :param backend:
        The backend to use.

    :param algorithm:
        The computational algorithm used for computing HGR.

    :param kwargs:
        Additional algorithm-specific arguments.
    """
    if algorithm == 'double-kernel':
        return DoubleKernelHGR(backend=backend, **kwargs)
    elif algorithm == 'single-kernel':
        return SingleKernelHGR(backend=backend, **kwargs)
    elif algorithm == 'adversarial':
        return AdversarialHGR(backend=backend, **kwargs)
    else:
        raise AssertionError(f"Unknown HGR algorithm '{algorithm}'")
