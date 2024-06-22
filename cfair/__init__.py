from typing import Union

from cfair.backends import Backend
from cfair.metrics.adversarial import AdversarialHGR
from cfair.metrics.kernel import DoubleKernelHGR, SingleKernelHGR, DoubleKernelGeDI, SingleKernelGeDI, \
    GeneralizedDisparateImpact
from cfair.metrics.metric import Metric


def hgr(backend: Union[str, Backend], algorithm: str, **kwargs) -> Metric:
    """Builds a Hirschfield-Gebelin-Renyi (HGR) indicator instance.

    :param backend:
        The backend to use.

    :param algorithm:
        The computational algorithm used for computing the metric.

    :param kwargs:
        Additional algorithm-specific arguments.
    """
    algorithm = algorithm.lower().replace('-', ' ').replace('_', ' ')
    if algorithm == 'double kernel':
        return DoubleKernelHGR(backend=backend, **kwargs)
    elif algorithm == 'single kernel':
        return SingleKernelHGR(backend=backend, **kwargs)
    elif algorithm == 'adversarial':
        return AdversarialHGR(backend=backend, **kwargs)
    else:
        raise AssertionError(f"Unsupported HGR algorithm '{algorithm}'")


def gedi(backend: Union[str, Backend], algorithm: str = 'default', **kwargs) -> None:
    """Builds a Generalized Disparate Impact (GeDI) indicator instance.

    :param backend:
        The backend to use.

    :param algorithm:
        The computational algorithm used for computing the metric.

    :param kwargs:
        Additional algorithm-specific arguments.
    """
    algorithm = algorithm.lower().replace('-', ' ').replace('_', ' ')
    if algorithm == 'default':
        return GeneralizedDisparateImpact(backend=backend, **kwargs)
    elif algorithm == 'double kernel':
        return DoubleKernelGeDI(backend=backend, **kwargs)
    elif algorithm == 'single kernel':
        return SingleKernelGeDI(backend=backend, **kwargs)
    else:
        raise AssertionError(f"Unsupported GeDI algorithm '{algorithm}'")


def metric(backend: Union[str, Backend], indicator: str, algorithm: str, **kwargs) -> Metric:
    """Builds the instance of a fairness metric for continuous attributes using the given indicator semantics.

    :param backend:
        The backend to use.

    :param indicator:
        The type of indicator semantics, either 'hgr' or 'gedi'.

    :param algorithm:
        The computational algorithm used for computing the metric.

    :param kwargs:
        Additional algorithm-specific arguments.
    """
    indicator = indicator.lower()
    if indicator == 'hgr':
        return hgr(backend=backend, algorithm=algorithm, **kwargs)
    elif indicator == 'gedi':
        return gedi(backend=backend, algorithm=algorithm, **kwargs)
    else:
        raise AssertionError(f"Unsupported indicator '{indicator}'")
