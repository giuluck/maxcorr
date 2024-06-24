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
from scipy.optimize import minimize

from cfair.backends import Backend
from cfair.metrics.kernel.abstract import KernelBasedMetric, DoubleKernelMetric, SingleKernelMetric


class KernelBasedGeDI(KernelBasedMetric, ABC):
    """Kernel-based metric interface where the computed indicator is GeDI."""

    def _indicator(self, f, g, a0: Optional, b0: Optional) -> Tuple[Any, np.ndarray, np.ndarray]:
        n, degree_a = self.backend.shape(f)
        _, degree_b = self.backend.shape(g)
        # if both degrees are 1 compute the explicit indicator and return it along with alpha = [gedi] and beta = [1]
        if degree_a == 1 and degree_b == 1:
            f = self.backend.reshape(f, shape=-1)
            g = self.backend.reshape(g, shape=-1)
            var_f, cov_fg = self.backend.cov(f, g)[0]
            gedi = self.backend.abs(cov_fg) / (var_f + self.eps)
            return gedi, np.ones(1), np.ones(1)
        # if both degrees are higher than 1, return a not-implemented error
        if degree_a > 1 and degree_b > 1:
            raise NotImplementedError("GeDI formulation is currently admissible for one kernel at a time.")
        # otherwise, swap the vectors so that the degree 1 is on the second one, then solve the problem either via
        # lstsq computing gedi using the absolute value, or via nonlinear_lstsq using || alpha ||_1 == 1 as constraint
        f_swap, g_swap, x0, degree, swap = (f, g, a0, degree_a, False) if degree_b == 1 else (g, f, b0, degree_b, True)
        if self.use_lstsq:
            alpha_tilde = self.backend.lstsq(A=f_swap, b=self.backend.reshape(g_swap, shape=-1))
            gedi = self.backend.sum(self.backend.abs(alpha_tilde))
            alpha_numpy = self.backend.numpy(alpha_tilde) / self.backend.numpy(gedi)
            beta_numpy = np.ones(1)
        else:
            f_numpy = self.backend.numpy(f_swap)
            g_numpy = self.backend.numpy(g_swap).reshape(-1)

            # define the function to optimize as the least square problem:
            #   - func: || F @ alpha - g ||_2^2 = (F @ alpha - g) @ (F @ alpha - g)
            #   - grad:  2 * F.T @ (F @ alpha - g)
            #   - hess:  2 * F.T @ F
            def _fun(alp):
                diff_numpy = f_numpy @ alp - g_numpy
                obj_func = diff_numpy @ diff_numpy
                obj_grad = 2 * f_numpy.T @ diff_numpy
                return obj_func  # , obj_grad

            # if no guess is provided, set the initial point as [ 1 ] then solve
            s = minimize(
                _fun,
                # jac=True,
                # hess=lambda *_: 2 * f_numpy.T @ f_numpy,
                x0=np.ones(degree) if x0 is None else x0,
                method=self.method,
                tol=self.tol,
                options={'maxiter': self.maxiter}
            )
            alpha_numpy = s.x / np.abs(s.x).sum()
            beta_numpy = np.ones(1)
            alpha = self.backend.cast(alpha_numpy, dtype=self.backend.dtype(f_swap))
            fa = self.backend.matmul(f_swap, alpha)
            gb = self.backend.reshape(g_swap, shape=-1)
            var_f, cov_fg = self.backend.cov(fa, gb)[0]
            gedi = self.backend.abs(cov_fg) / (var_f + self.eps)
        # re-swap alpha and beta if necessary
        alpha_numpy, beta_numpy = (beta_numpy, alpha_numpy) if swap else (alpha_numpy, beta_numpy)
        return gedi, alpha_numpy, beta_numpy


class GeneralizedDisparateImpact(DoubleKernelMetric, KernelBasedGeDI):
    """GeDI indicator computed as for its original formulation in the paper "Generalized Disparate Impact for
    Configurable Fairness Solutions in ML", i.e., using the kernel of the first vector only."""

    def __init__(self,
                 backend: Union[str, Backend] = 'numpy',
                 kernel: Union[int, Callable[[Any], list]] = 3,
                 method: str = 'trust-constr',
                 maxiter: int = 1000,
                 eps: float = 1e-9,
                 tol: float = 1e-9,
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


class SymmetricGeDI(SingleKernelMetric, KernelBasedGeDI):
    """Double-kernel GeDI indicator computed as max { GeDI(a, b; K(a), 1), GeDI(a, b; 1, K(b)) }"""
    pass
