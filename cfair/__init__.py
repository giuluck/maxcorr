from typing import Union

from cfair.backends import Backend
from cfair.indicators import DoubleKernelHGR, SingleKernelHGR
from cfair.indicators.indicator import Indicator, HGRIndicator, GeDIIndicator, NLCIndicator
from cfair.indicators.kernel import DoubleKernelGeDI, SingleKernelGeDI, DoubleKernelNLC, SingleKernelNLC
from cfair.indicators.neural import NeuralHGR, NeuralGeDI, NeuralNLC


def indicator(backend: Union[str, Backend], semantics: str, algorithm: str, **kwargs) -> Indicator:
    """Builds the instance of a fairness indicator for continuous attributes using the given indicator semantics.

    :param backend:
        The backend to use.

    :param semantics:
        The type of indicator semantics, either 'hgr' or 'gedi'.

    :param algorithm:
        The computational algorithm used for computing the indicator.

    :param kwargs:
        Additional algorithm-specific arguments.
    """
    semantics = semantics.lower()
    if semantics == 'hgr':
        return hgr(backend=backend, algorithm=algorithm, **kwargs)
    elif semantics == 'gedi':
        return gedi(backend=backend, algorithm=algorithm, **kwargs)
    elif semantics == 'nlc':
        return nlc(backend=backend, algorithm=algorithm, **kwargs)
    else:
        raise AssertionError(f"Unsupported semantics '{semantics}'")


def hgr(backend: Union[str, Backend], algorithm: str = 'dk', **kwargs) -> HGRIndicator:
    """Builds a Hirschfield-Gebelin-Renyi (HGR) indicator instance.

    :param backend:
        The backend to use.

    :param algorithm:
        The computational algorithm used for computing the indicator.

    :param kwargs:
        Additional algorithm-specific arguments.
    """
    algorithm = algorithm.lower().replace('-', ' ').replace('_', ' ')
    if algorithm in ['dk', 'double kernel']:
        return DoubleKernelHGR(backend=backend, **kwargs)
    elif algorithm in ['sk', 'single kernel']:
        return SingleKernelHGR(backend=backend, **kwargs)
    elif algorithm in ['nn', 'neural']:
        return NeuralHGR(backend=backend, **kwargs)
    else:
        raise AssertionError(f"Unsupported HGR algorithm '{algorithm}'")


def gedi(backend: Union[str, Backend], algorithm: str = 'dk', **kwargs) -> GeDIIndicator:
    """Builds a Generalized Disparate Impact (GeDI) indicator instance.

    :param backend:
        The backend to use.

    :param algorithm:
        The computational algorithm used for computing the indicator.

    :param kwargs:
        Additional algorithm-specific arguments.
    """
    algorithm = algorithm.lower().replace('-', ' ').replace('_', ' ')
    if algorithm in ['dk', 'double kernel']:
        return DoubleKernelGeDI(backend=backend, **kwargs)
    elif algorithm in ['sk', 'single kernel']:
        return SingleKernelGeDI(backend=backend, **kwargs)
    elif algorithm in ['nn', 'neural']:
        return NeuralGeDI(backend=backend, **kwargs)
    else:
        raise AssertionError(f"Unsupported GeDI algorithm '{algorithm}'")


def nlc(backend: Union[str, Backend], algorithm: str = 'dk', **kwargs) -> NLCIndicator:
    """Builds a Non-Linear Covariance (NLC) indicator instance.

    :param backend:
        The backend to use.

    :param algorithm:
        The computational algorithm used for computing the indicator.

    :param kwargs:
        Additional algorithm-specific arguments.
    """
    algorithm = algorithm.lower().replace('-', ' ').replace('_', ' ')
    if algorithm in ['dk', 'double kernel']:
        return DoubleKernelNLC(backend=backend, **kwargs)
    elif algorithm in ['sk', 'single kernel']:
        return SingleKernelNLC(backend=backend, **kwargs)
    elif algorithm in ['nn', 'neural']:
        return NeuralNLC(backend=backend, **kwargs)
    else:
        raise AssertionError(f"Unsupported GeDI algorithm '{algorithm}'")
