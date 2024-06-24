"""
GeDI and HGR implementations of the method from "Generalized Disparate Impact for Configurable Fairness Solutions in ML"
by Luca Giuliani, Eleonora Misino and Michele Lombardi, and "Enhancing the Applicability of Fair Learning with
Continuous Attributes" by Luca Giuliani and Michele Lombardi, respectively. The code has been partially taken and
reworked from the repositories containing the code of the paper, respectively:
- https://github.com/giuluck/GeneralizedDisparateImpact/tree/main
- https://github.com/giuluck/kernel-based-hgr/tree/main
"""

from abc import ABC
from typing import Optional

import numpy as np
from scipy.optimize import NonlinearConstraint

from cfair.metrics.kernel.abstract import KernelBasedMetric, DoubleKernelMetric, SingleKernelMetric


class KernelBasedHGR(KernelBasedMetric, ABC):
    """Kernel-based metric interface where the computed indicator is HGR."""

    def _indicator(self, f, g, a0: Optional, b0: Optional) -> KernelBasedMetric.Result:
        n, degree_a = self.backend.shape(f)
        _, degree_b = self.backend.shape(g)
        # handle trivial or simpler cases:
        #  - if both degrees are 1, simply compute the projected vectors as standardized original vectors
        #  - if one degree is 1, standardize that vector and compute the other's coefficients using lstsq
        #  - if no degree is 1, use the nonlinear lstsq optimization routine via scipy.optimize
        alpha = self.backend.ones(1, dtype=self.backend.dtype(f))
        beta = self.backend.ones(1, dtype=self.backend.dtype(g))
        alpha_numpy, beta_numpy = np.ones(1), np.ones(1)
        if degree_a == 1 and degree_b == 1:
            pass
        elif degree_a == 1 and self.use_lstsq:
            f = self.backend.standardize(f, eps=self.eps)
            beta = self.backend.lstsq(A=g, b=self.backend.reshape(f, shape=-1))
            beta_numpy = self.backend.numpy(beta)
        elif degree_b == 1 and self.use_lstsq:
            g = self.backend.standardize(g, eps=self.eps)
            alpha = self.backend.lstsq(A=f, b=self.backend.reshape(g, shape=-1))
            alpha_numpy = self.backend.numpy(alpha)
        else:
            f_numpy = self.backend.numpy(f)
            g_numpy = self.backend.numpy(g)
            # define the constraint
            #   - func:   var(G @ beta) --> = 1
            #   - grad: [ 0 | 2 * G.T @ G @ beta / n ]
            #   - hess: [ 0 |         0       ]
            #           [ 0 | 2 * G.T @ G / n ]
            cst_hess = np.zeros(shape=(degree_a + degree_b, degree_a + degree_b), dtype=float)
            cst_hess[degree_a:, degree_a:] = 2 * g_numpy.T @ g_numpy / n
            cst = NonlinearConstraint(
                fun=lambda inp: np.var(g_numpy @ inp[degree_a:], ddof=0),
                jac=lambda inp: np.concatenate(([0] * degree_a, 2 * g_numpy.T @ g_numpy @ inp[degree_a:] / n)),
                hess=lambda *_: cst_hess,
                lb=1,
                ub=1
            )
            alpha_numpy, beta_numpy = self._constrained_lstsq(f=f_numpy, g=g_numpy, a0=a0, b0=b0, constraints=[cst])
            alpha = self.backend.cast(alpha_numpy, dtype=self.backend.dtype(f))
            beta = self.backend.cast(beta_numpy, dtype=self.backend.dtype(g))
        # return the HGR value as the absolute value of the (mean) vector product (since the vectors are standardized)
        fa = self.backend.standardize(self.backend.matmul(f, alpha), eps=self.eps)
        gb = self.backend.standardize(self.backend.matmul(g, beta), eps=self.eps)
        correlation = self.backend.matmul(fa, gb) / n
        return self.backend.abs(correlation), alpha_numpy, beta_numpy


class DoubleKernelHGR(DoubleKernelMetric, KernelBasedHGR):
    """HGR indicator computed using two different explicit kernels for the variables."""
    pass


class SingleKernelHGR(SingleKernelMetric, KernelBasedHGR):
    """HGR indicator computed using a single kernel for the variables, then taking the maximal value."""
    pass
