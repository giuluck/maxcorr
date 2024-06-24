"""
GeDI and HGR implementations of the method from "Generalized Disparate Impact for Configurable Fairness Solutions in ML"
by Luca Giuliani, Eleonora Misino and Michele Lombardi, and "Enhancing the Applicability of Fair Learning with
Continuous Attributes" by Luca Giuliani and Michele Lombardi, respectively. The code has been partially taken and
reworked from the repositories containing the code of the paper, respectively:
- https://github.com/giuluck/GeneralizedDisparateImpact/tree/main
- https://github.com/giuluck/kernel-based-hgr/tree/main
"""

from abc import ABC
from typing import Tuple, Optional, Any, Union, Callable

import numpy as np

from cfair.backends import Backend
from cfair.metrics.kernel.abstract import KernelBasedMetric, DoubleKernelMetric


class KernelBasedGeDI(KernelBasedMetric, ABC):
    """Kernel-based metric interface where the computed indicator is GeDI."""

    def _indicator(self, f, g, a0: Optional, b0: Optional) -> Tuple[Any, np.ndarray, np.ndarray]:
        pass


class GeneralizedDisparateImpact(DoubleKernelMetric, KernelBasedGeDI):
    """GeDI indicator computed as for its original formulation in the paper "Generalized Disparate Impact for
    Configurable Fairness Solutions in ML", i.e., using the kernel of the first vector only."""

    def __init__(self,
                 backend: Union[str, Backend] = 'numpy',
                 kernel: Union[int, Callable[[Any], list]] = 3,
                 method: str = 'trust-constr',
                 maxiter: int = 1000,
                 eps: float = 1e-9,
                 tol: float = 1e-2,
                 use_lstsq: bool = True,
                 delta_independent: Optional[float] = None):
        """
        :param backend:
            The backend to use to compute the metric, or its alias.

        :param kernel:
            Either a callable f(a) representing the input kernel, or an integer degree for a polynomial kernel.

        :param method:
            The optimization method as in scipy.optimize.minimize, either 'trust-constr' or 'SLSQP'.

        :param maxiter:
            The maximal number of iterations before stopping the optimization process as in scipy.optimize.minimize.

        :param eps:
            The epsilon value used to avoid division by zero in case of null standard deviation.

        :param tol:
            The tolerance used in the stopping criterion for the optimization process scipy.optimize.minimize.

        :param use_lstsq:
            Whether to rely on the least-square problem closed-form solution when at least one of the degrees is 1.

        :param delta_independent:
            A delta value used to select linearly dependent columns and remove them, or None to avoid this step.
        """
        super(GeneralizedDisparateImpact, self).__init__(
            backend=backend,
            kernel_a=kernel,
            kernel_b=1,
            method=method,
            maxiter=maxiter,
            eps=eps,
            tol=tol,
            use_lstsq=use_lstsq,
            delta_independent=delta_independent
        )
