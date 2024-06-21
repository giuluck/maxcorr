from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Any, Union

import numpy as np
import scipy
from scipy.optimize import NonlinearConstraint, minimize

from cfair.backend import Backend
from cfair.hgr.hgr import HGR


class KernelBasedHGR(HGR):
    """Kernel-based HGR interface."""

    @dataclass(frozen=True, init=True, repr=False, eq=False, unsafe_hash=None)
    class Result(HGR.Result):
        """Data class representing the results of a KernelBasedHGR computation."""

        alpha: np.ndarray = field()
        """The coefficient vector for the f copula transformation."""

        beta: np.ndarray = field()
        """The coefficient vector for the f copula transformation."""

    def __init__(self,
                 backend: Union[str, Backend],
                 method: str,
                 maxiter: int,
                 eps: float,
                 tol: float,
                 use_lstsq: bool,
                 delta: float,
                 lasso: float):
        """
        :param backend:
            The backend to use to compute the HGR correlation, or its alias.

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

        :param delta:
            A delta value used to decide whether two columns are linearly dependent.

        :param lasso:
            The amount of lasso regularization introduced when computing HGR.
        """
        super(KernelBasedHGR, self).__init__(backend=backend)
        self._method: str = method
        self._maxiter: int = maxiter
        self._eps: float = eps
        self._tol: float = tol
        self._use_lstsq: bool = use_lstsq
        self._delta: float = delta
        self._lasso: lasso = lasso

    @property
    @abstractmethod
    def degree_a(self) -> int:
        """The kernel degree for the first variable."""
        pass

    @property
    @abstractmethod
    def degree_b(self) -> int:
        """The kernel degree for the second variable."""
        pass

    @property
    def method(self) -> str:
        """The optimization method as in scipy.optimize.minimize, either 'trust-constr' or 'SLSQP'."""
        return self._method

    @property
    def maxiter(self) -> int:
        """The maximal number of iterations before stopping the optimization process as in scipy.optimize.minimize."""
        return self._maxiter

    @property
    def eps(self) -> float:
        """The epsilon value used to avoid division by zero in case of null standard deviation."""
        return self._eps

    @property
    def tol(self) -> float:
        """The tolerance used in the stopping criterion for the optimization process scipy.optimize.minimize."""
        return self._tol

    @property
    def use_lstsq(self) -> bool:
        """Whether to rely on the least-square problem closed-form solution when at least one of the degrees is 1."""
        return self._use_lstsq

    @property
    def delta(self) -> float:
        """A delta value used to decide whether two columns are linearly dependent."""
        return self._delta

    @property
    def lasso(self) -> float:
        """The amount of lasso regularization introduced when computing HGR."""
        return self._lasso

    def _f(self, a) -> Any:
        fa = self.backend.stack([a ** d - self.backend.mean(a ** d) for d in np.arange(self.degree_a) + 1])
        # noinspection PyUnresolvedReferences
        alpha = self.backend.cast(self.last_result.alpha)
        return self.backend.matmul(fa, alpha)

    def _g(self, b) -> Any:
        gb = self.backend.stack([b ** d - self.backend.mean(b ** d) for d in np.arange(self.degree_b) + 1])
        # noinspection PyUnresolvedReferences
        beta = self.backend.cast(self.last_result.beta)
        return self.backend.matmul(gb, beta)

    def _get_linearly_independent(self, f: np.ndarray, g: np.ndarray) -> Tuple[List[int], List[int]]:
        """Returns the list of indices of those columns that are linearly independent to other ones."""
        n, dx = f.shape
        _, dy = g.shape
        d = dx + dy
        # build a new matrix [ 1 | F_1 | G_1 | F_2 | G_2 | ... ]
        #   - this order is chosen so that lower grades are preferred in case of linear dependencies
        #   - the F and G indices are built depending on which kernel has the higher degree
        if dx < dy:
            f_indices = [2 * i + 1 for i in range(dx)]
            g_indices = [2 * i + 2 for i in range(dx)] + [i + 1 for i in range(2 * dx, d)]
        else:
            f_indices = [2 * i + 1 for i in range(dy)] + [i + 1 for i in range(2 * dy, d)]
            g_indices = [2 * i + 2 for i in range(dy)]
        fg_bias = np.ones((len(f), d + 1))
        fg_bias[:, f_indices] = f
        fg_bias[:, g_indices] = g
        # compute the QR factorization and retrieve the R matrix
        #   - get the diagonal of R
        #   - if a null value is found, it means that the respective column is linearly dependent to other columns
        # noinspection PyUnresolvedReferences
        r = scipy.linalg.qr(fg_bias, mode='r')[0]
        r = np.abs(np.diag(r))
        # eventually, retrieve the indices to be set to zero:
        #   - create a range going from 0 to degree - 1
        #   - mask it by selecting all those value in the diagonal that are smaller than the tolerance
        #   - finally exclude the first value in both cases since their linear dependence might be caused by a
        #      deterministic dependency in the data which we don't want to exclude
        f_indices = np.arange(dx)[r[f_indices] <= self.delta][1:]
        g_indices = np.arange(dy)[r[g_indices] <= self.delta][1:]
        return [idx for idx in range(dx) if idx not in f_indices], [idx for idx in range(dy) if idx not in g_indices]

    def _higher_order_coefficients(self,
                                   f: np.ndarray,
                                   g: np.ndarray,
                                   a0: Optional[np.ndarray],
                                   b0: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the kernel-based hgr for higher order degrees."""
        degree_x, degree_y = f.shape[1], g.shape[1]
        # retrieve the indices of the linearly dependent columns and impose a linear constraint so that the respective
        # weight is null for all but the first one (this processing step allow to avoid degenerate cases when the
        # matrix is not full rank)
        f_indices, g_indices = self._get_linearly_independent(f=f, g=g)
        f_slim = f[:, f_indices]
        g_slim = g[:, g_indices]
        n, dx = f_slim.shape
        _, dy = g_slim.shape
        d = dx + dy
        fg = np.concatenate((f_slim, -g_slim), axis=1)

        # define the function to optimize as the least square problem:
        #   - func:   || F @ alpha - G @ beta ||_2^2 =
        #           =   (F @ alpha - G @ beta) @ (F @ alpha - G @ beta)
        #   - grad:   [ 2 * F.T @ (F @ alpha - G @ beta) | -2 * G.T @ (F @ alpha - G @ beta) ] =
        #           =   2 * [F | -G].T @ (F @ alpha - G @ beta)
        #   - hess:   [  2 * F.T @ F | -2 * F.T @ G ]
        #             [ -2 * G.T @ F |  2 * G.T @ G ] =
        #           =    2 * [F  -G].T @ [F  -G]
        #
        # plus, add the lasso penalizer
        #   - func:     norm_1([alpha, beta])
        #   - grad:   [ sign(alpha) | sign(beta) ]
        #   - hess:   [      0      |      0     ]
        #             [      0      |      0     ]
        def _fun(inp):
            alp, bet = inp[:dx], inp[dx:]
            diff = f_slim @ alp - g_slim @ bet
            obj_func = diff @ diff
            obj_grad = 2 * fg.T @ diff
            pen_func = np.abs(inp).sum()
            pen_grad = np.sign(inp)
            # noinspection PyUnresolvedReferences
            return obj_func + self.lasso * pen_func, obj_grad + self.lasso * pen_grad

        fun_hess = 2 * fg.T @ fg

        # define the constraint
        #   - func:   var(G @ beta) --> = 1
        #   - grad: [ 0 | 2 * G.T @ G @ beta / n ]
        #   - hess: [ 0 |         0       ]
        #           [ 0 | 2 * G.T @ G / n ]
        cst_hess = np.zeros(shape=(d, d), dtype=float)
        cst_hess[dx:, dx:] = 2 * g_slim.T @ g_slim / n
        constraint = NonlinearConstraint(
            fun=lambda inp: np.var(g_slim @ inp[dx:], ddof=0),
            jac=lambda inp: np.concatenate(([0] * dx, 2 * g_slim.T @ g_slim @ inp[dx:] / n)),
            hess=lambda *_: cst_hess,
            lb=1,
            ub=1
        )
        # if no guess is provided, set the initial point as [ 1 / std(F @ 1) | 1 / std(G @ 1) ] then solve the problem
        a0 = np.ones(dx) / np.sqrt(f_slim.sum(axis=1).var(ddof=0) + self.eps) if a0 is None else a0[f_indices]
        b0 = np.ones(dy) / np.sqrt(g_slim.sum(axis=1).var(ddof=0) + self.eps) if b0 is None else b0[g_indices]
        x0 = np.concatenate((a0, b0))
        s = minimize(
            _fun,
            jac=True,
            hess=lambda *_: fun_hess,
            x0=x0,
            constraints=[constraint],
            method=self.method,
            tol=self.tol,
            options={'maxiter': self.maxiter}
        )
        # reconstruct alpha and beta by adding zeros wherever the indices were not considered
        alpha = np.zeros(degree_x)
        alpha[f_indices] = s.x[:dx]
        beta = np.zeros(degree_y)
        beta[g_indices] = s.x[dx:]
        return alpha, beta

    def _kbhgr(self, a, b, degree_a: int = 1, degree_b: int = 1, a0: Optional = None, b0: Optional = None) -> Result:
        """Computes HGR using numpy as backend and returns the correlation (without alpha and beta)."""
        backend = self.backend
        # build the kernel matrices
        f = backend.stack([a ** d - backend.mean(a ** d) for d in np.arange(self.degree_a) + 1])
        g = backend.stack([b ** d - backend.mean(b ** d) for d in np.arange(self.degree_b) + 1])
        # handle trivial or simpler cases:
        #  - if both degrees are 1, simply compute the projected vectors as standardized original vectors
        #  - if one degree is 1, standardize that vector and compute the other's coefficients using lstsq
        #  - if no degree is 1, use the optimization routine and compute the projected vectors from the coefficients
        alpha, beta = np.ones(1), np.ones(1)
        if degree_a == 1 and degree_b == 1:
            fa = backend.standardize(a, eps=self.eps)
            gb = backend.standardize(b, eps=self.eps)
        elif degree_a == 1 and self.use_lstsq:
            fa = backend.standardize(a, eps=self.eps)
            beta = backend.lstsq(g, fa)
            gb = backend.standardize(backend.matmul(g, beta), eps=self.eps)
            beta = backend.numpy(beta)
        elif degree_b == 1 and self.use_lstsq:
            gb = backend.standardize(b, eps=self.eps)
            alpha = backend.lstsq(f, gb)
            fa = backend.standardize(backend.matmul(f, alpha), eps=self.eps)
            alpha = backend.numpy(alpha)
        else:
            alpha, beta = self._higher_order_coefficients(f=backend.numpy(f), g=backend.numpy(g), a0=a0, b0=b0)
            fa = backend.standardize(backend.matmul(f, backend.cast(alpha, dtype=backend.dtype(f))), eps=self.eps)
            gb = backend.standardize(backend.matmul(g, backend.cast(beta, dtype=backend.dtype(g))), eps=self.eps)
        # return the correlation as the absolute value of the (mean) vector product (since the vectors are standardized)
        correlation = backend.abs(backend.matmul(fa, gb) / backend.len(fa))
        return KernelBasedHGR.Result(
            a=a,
            b=b,
            correlation=correlation,
            num_call=self.num_calls,
            hgr=self,
            alpha=alpha,
            beta=beta,
        )


class DoubleKernelHGR(KernelBasedHGR):
    """Kernel-based HGR computed by solving a constrained least square problem using a minimization solver."""

    def __init__(self,
                 backend: Union[str, Backend] = 'numpy',
                 degree_a: int = 3,
                 degree_b: int = 3,
                 method: str = 'trust-constr',
                 maxiter: int = 1000,
                 eps: float = 1e-9,
                 tol: float = 1e-2,
                 use_lstsq: bool = True,
                 delta: float = 1e-2,
                 lasso: float = 0.0):
        """
        :param backend:
            The backend to use to compute the HGR correlation, or its alias.

        :param degree_a:
            The kernel degree for the first variable.

        :param degree_b:
            The kernel degree for the second variable.

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

        :param delta:
            A delta value used to decide whether two columns are linearly dependent.

        :param lasso:
            The amount of lasso regularization introduced when computing HGR.
        """
        super(DoubleKernelHGR, self).__init__(
            backend=backend,
            method=method,
            maxiter=maxiter,
            eps=eps,
            tol=tol,
            use_lstsq=use_lstsq,
            delta=delta,
            lasso=lasso
        )
        self._degree_a: int = degree_a
        self._degree_b: int = degree_b

    @property
    def degree_a(self) -> int:
        return self._degree_a

    @property
    def degree_b(self) -> int:
        return self._degree_b

    def _compute(self, a, b) -> KernelBasedHGR.Result:
        # noinspection PyUnresolvedReferences
        a0, b0 = (None, None) if self.last_result is None else (self.last_result.alpha, self.last_result.beta)
        return self._kbhgr(a=a, b=b, degree_a=self.degree_a, degree_b=self.degree_b, a0=a0, b0=b0)


class SingleKernelHGR(KernelBasedHGR):
    """Kernel-based HGR computed using one kernel only for both variables and then taking the maximal correlation."""

    def __init__(self,
                 backend: Union[str, Backend] = 'numpy',
                 degree: int = 3,
                 method: str = 'trust-constr',
                 maxiter: int = 1000,
                 eps: float = 1e-9,
                 tol: float = 1e-2,
                 use_lstsq: bool = True,
                 delta: float = 1e-2,
                 lasso: float = 0.0):
        """
        :param backend:
            The backend to use to compute the HGR correlation, or its alias.

        :param degree:
            The kernel degree for the variables.

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

        :param delta:
            A delta value used to decide whether two columns are linearly dependent.

        :param lasso:
            The amount of lasso regularization introduced when computing HGR.
        """
        super(SingleKernelHGR, self).__init__(
            backend=backend,
            method=method,
            maxiter=maxiter,
            eps=eps,
            tol=tol,
            use_lstsq=use_lstsq,
            delta=delta,
            lasso=lasso
        )
        self._degree: int = degree

    @property
    def degree(self) -> int:
        """The kernel degree for the variables."""
        return self._degree

    @property
    def degree_a(self) -> int:
        return self._degree

    @property
    def degree_b(self) -> int:
        return self._degree

    def _compute(self, a, b) -> KernelBasedHGR.Result:
        # noinspection PyUnresolvedReferences
        a0, b0 = (None, None) if self.last_result is None else (self.last_result.alpha, self.last_result.beta)
        res_a = self._kbhgr(a=a, b=b, degree_a=self.degree, a0=a0)
        res_b = self._kbhgr(a=a, b=b, degree_b=self.degree, b0=b0)
        cor_a = res_a.correlation
        cor_b = res_b.correlation
        if cor_a > cor_b:
            correlation = cor_a
            alpha = res_a.alpha
            beta = np.concatenate((res_a.beta, np.zeros(self.degree - 1)))
        else:
            correlation = cor_b
            beta = res_b.beta
            alpha = np.concatenate((res_b.alpha, np.zeros(self.degree - 1)))
        return KernelBasedHGR.Result(
            a=a,
            b=b,
            correlation=correlation,
            num_call=self.num_calls,
            hgr=self,
            alpha=alpha,
            beta=beta,
        )
