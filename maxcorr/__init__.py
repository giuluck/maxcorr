from typing import Union

from maxcorr.backends import Backend
from maxcorr.indicators import Indicator, DoubleKernelIndicator, SingleKernelIndicator, NeuralIndicator, \
    LatticeIndicator, DensityIndicator, RandomizedIndicator
from maxcorr.typing import BackendType, SemanticsType, AlgorithmType


def indicator(semantics: SemanticsType = 'hgr',
              algorithm: AlgorithmType = 'dk',
              backend: Union[Backend, BackendType] = 'numpy',
              **kwargs) -> Indicator:
    """Builds the instance of an indicator for continuous attributes using the given indicator semantics.

    :param semantics:
        The type of indicator semantics.

    :param algorithm:
        The computational algorithm used for computing the indicator.

    :param backend:
        The backend to use, or its alias.

    :param kwargs:
        Additional algorithm-specific arguments.
    """
    algorithm = algorithm.lower().replace('-', ' ').replace('_', ' ')
    if algorithm in ['dk', 'double kernel']:
        return DoubleKernelIndicator(backend=backend, semantics=semantics, **kwargs)
    elif algorithm in ['sk', 'single kernel']:
        return SingleKernelIndicator(backend=backend, semantics=semantics, **kwargs)
    elif algorithm in ['nn', 'neural']:
        return NeuralIndicator(backend=backend, semantics=semantics, **kwargs)
    elif algorithm in ['kde', 'density']:
        return DensityIndicator(backend=backend, semantics=semantics, **kwargs)
    elif algorithm in ['rdc', 'randomized']:
        return RandomizedIndicator(backend=backend, semantics=semantics, **kwargs)
    elif algorithm in ['lat', 'lattice']:
        return LatticeIndicator(backend=backend, semantics=semantics, **kwargs)
    else:
        raise AssertionError(f"Unsupported algorithm '{algorithm}'")


def hgr(algorithm: AlgorithmType = 'dk', backend: Union[Backend, BackendType] = 'numpy', **kwargs) -> Indicator:
    """Builds a Hirschfield-Gebelin-Rényi (HGR) indicator instance.

    :param algorithm:
        The computational algorithm used for computing the indicator.

    :param backend:
        The backend to use, or its alias.

    :param kwargs:
        Additional algorithm-specific arguments.
    """
    return indicator(semantics='hgr', algorithm=algorithm, backend=backend, **kwargs)


def gedi(algorithm: AlgorithmType = 'dk', backend: Union[Backend, BackendType] = 'numpy', **kwargs) -> Indicator:
    """Builds a Generalized Disparate Impact (GeDI) indicator instance.

    :param algorithm:
        The computational algorithm used for computing the indicator.

    :param backend:
        The backend to use, or its alias.

    :param kwargs:
        Additional algorithm-specific arguments.
    """
    return indicator(semantics='gedi', algorithm=algorithm, backend=backend, **kwargs)


def nlc(algorithm: AlgorithmType = 'dk', backend: Union[Backend, BackendType] = 'numpy', **kwargs) -> Indicator:
    """Builds a Non-Linear Covariance (NLC) indicator instance.

    :param algorithm:
        The computational algorithm used for computing the indicator.

    :param backend:
        The backend to use, or its alias.

    :param kwargs:
        Additional algorithm-specific arguments.
    """
    return indicator(semantics='nlc', algorithm=algorithm, backend=backend, **kwargs)
