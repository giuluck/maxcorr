"""
GeDI and HGR implementations of the method from "Generalized Disparate Impact for Configurable Fairness Solutions in ML"
by Luca Giuliani, Eleonora Misino and Michele Lombardi, and "Enhancing the Applicability of Fair Learning with
Continuous Attributes" by Luca Giuliani and Michele Lombardi, respectively. The code has been partially taken and
reworked from the repositories containing the code of the paper, respectively:
- https://github.com/giuluck/GeneralizedDisparateImpact/tree/main
- https://github.com/giuluck/kernel-based-hgr/tree/main
"""

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Any, Union, Callable, final

import numpy as np
import scipy
from scipy.optimize import minimize, NonlinearConstraint

from cfair.backends import Backend
from cfair.metrics.metric import CopulaMetric


class KernelBasedMetric(CopulaMetric):
    """Kernel-based metric computed using user-defined kernels to approximate the copula transformations."""

    @dataclass(frozen=True, init=True, repr=False, eq=False, unsafe_hash=None)
    class Result(CopulaMetric.Result):
        """Data class representing the results of a KernelBasedMetric computation."""

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
                 delta_independent: Optional[float]):
        """
        :param backend:
            The backend to use to compute the metric, or its alias.

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
        super(KernelBasedMetric, self).__init__(backend=backend)
        self._method: str = method
        self._maxiter: int = maxiter
        self._eps: float = eps
        self._tol: float = tol
        self._use_lstsq: bool = use_lstsq
        self._delta_independent: Optional[float] = delta_independent

    @abstractmethod
    def kernel_a(self, a) -> list:
        """The list of kernels for the first variable."""
        pass

    @abstractmethod
    def kernel_b(self, b) -> list:
        """The list of kernels for the second variable."""
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
    def delta_independent(self) -> Optional[float]:
        """A delta value used to select linearly dependent columns and remove them, or None to avoid this step."""
        return self._delta_independent

    @final
    def _f(self, a) -> Any:
        kernel = self.backend.stack(self.kernel_a(a), axis=1)
        # noinspection PyUnresolvedReferences
        alpha = self.backend.cast(self.last_result.alpha)
        fa = self.backend.matmul(kernel, alpha)
        return self.backend.standardize(fa, eps=self.eps)

    @final
    def _g(self, b) -> Any:
        kernel = self.backend.stack(self.kernel_b(b), axis=1)
        # noinspection PyUnresolvedReferences
        beta = self.backend.cast(self.last_result.beta)
        gb = self.backend.matmul(kernel, beta)
        return self.backend.standardize(gb, eps=self.eps)

    @final
    def _indices(self, f: list, g: list) -> Tuple[Tuple[list, List[int]], Tuple[list, List[int]]]:
        def independent(m):
            # add the bias to the matrix
            b = np.ones(shape=(len(m), 1))
            m = np.concatenate((b, m), axis=1)
            # compute the QR factorization
            r = scipy.linalg.qr(m, mode='r')[0]
            # build the diagonal of the R matrix (excluding the bias column)
            r = np.abs(np.diag(r)[1:])
            # independent columns are those having a value higher than the tolerance
            mask = r >= self.delta_independent
            # handle the case in which all constant vectors are passed (return at least one column as independent)
            if not np.any(mask):
                mask[0] = True
            return mask

        # if the linear dependencies removal step is not selected, simply return all indices
        if self.delta_independent is None:
            return (f, list(range(len(f)))), (g, list(range(len(g))))
        # otherwise find independent columns for f and g, respectively
        f_numpy = np.stack([self.backend.numpy(fi) for fi in f], axis=1)
        f_independent = np.arange(f_numpy.shape[1])[independent(f_numpy)]
        g_numpy = np.stack([self.backend.numpy(gi) for gi in g], axis=1)
        g_independent = np.arange(g_numpy.shape[1])[independent(g_numpy)]
        # build the joint kernel matrix (with dependent columns removed) as [ 1 | F_1 | G_1 | F_2 | G_2 | ... ]
        #   - this order is chosen so that lower grades are preferred in case of linear dependencies
        #   - the F and G indices are built depending on which kernel has the higher degree
        da = len(f_independent)
        db = len(g_independent)
        d = da + db
        if da < db:
            f_indices = np.array([2 * i for i in range(da)])
            g_indices = np.array([2 * i + 1 for i in range(da)] + list(range(2 * da, d)))
        else:
            f_indices = np.array([2 * i for i in range(db)] + list(range(2 * db, d)))
            g_indices = np.array([2 * i + 1 for i in range(db)])
        # find independent columns the joint matrix in case of co-dependencies
        fg_numpy = np.zeros((len(f_numpy), d))
        fg_numpy[:, f_indices] = f_numpy[:, f_independent]
        fg_numpy[:, g_indices] = g_numpy[:, g_independent]
        fg_independent = independent(fg_numpy)
        # create lists of columns to exclude (dependent) rather than columns to include (independent) so to avoid
        # considering the first dependent values for each of the two matrix since their linear dependence might be
        # caused by a deterministic dependency in the data which would result in a maximal correlation
        f_excluded = f_independent[~fg_independent[f_indices]][1:]
        g_excluded = g_independent[~fg_independent[g_indices]][1:]
        # find the final columns and indices by selecting those independent values that were not excluded later
        f_indices, f_columns = [], []
        g_indices, g_columns = [], []
        for idx in f_independent:
            if idx not in f_excluded:
                f_indices.append(idx)
                f_columns.append(f[idx])
        for idx in g_independent:
            if idx not in g_excluded:
                g_indices.append(idx)
                g_columns.append(g[idx])
        return (f_columns, f_indices), (g_columns, g_indices)

    @final
    def _result(self, a, b, kernel_a: bool, kernel_b: bool, a0: Optional, b0: Optional) -> Result:
        # build the kernel matrices, compute their original degrees, and get the linearly independent indices
        f = self.kernel_a(a) if kernel_a else [a]
        g = self.kernel_b(b) if kernel_b else [b]
        degree_a, degree_b = len(f), len(g)
        (f, f_indices), (g, g_indices) = self._indices(f=f, g=g)
        # compute the original degrees of the matrices
        f = self.backend.stack(f, axis=1)
        f = f - self.backend.mean(f, axis=0)
        g = self.backend.stack(g, axis=1)
        g = g - self.backend.mean(g, axis=0)
        # compute the indicator value and the coefficients using the slim matrices
        val, alp, bet = self._indicator(
            f=f,
            g=g,
            a0=None if a0 is None else a0[f_indices],
            b0=None if b0 is None else b0[f_indices]
        )
        # reconstruct alpha and beta by adding zeros for the ignored indices
        alpha = np.zeros(degree_a)
        alpha[f_indices] = alp
        beta = np.zeros(degree_b)
        beta[g_indices] = bet
        # return the result instance
        return KernelBasedMetric.Result(
            a=a,
            b=b,
            value=val,
            num_call=self.num_calls,
            metric=self,
            alpha=alpha,
            beta=beta,
        )

    def _constrained_lstsq(self,
                           f: np.ndarray,
                           g: np.ndarray,
                           a0: Optional[np.ndarray],
                           b0: Optional[np.ndarray],
                           constraints: List[NonlinearConstraint]) -> Tuple[np.ndarray, np.ndarray]:
        n, degree_a = f.shape
        _, degree_b = g.shape
        fg = np.concatenate((f, -g), axis=1)

        # define the function to optimize as the least square problem:
        #   - func:   || F @ alpha - G @ beta ||_2^2 =
        #           =   (F @ alpha - G @ beta) @ (F @ alpha - G @ beta)
        #   - grad:   [ 2 * F.T @ (F @ alpha - G @ beta) | -2 * G.T @ (F @ alpha - G @ beta) ] =
        #           =   2 * [F | -G].T @ (F @ alpha - G @ beta)
        #   - hess:   [  2 * F.T @ F | -2 * F.T @ G ]
        #             [ -2 * G.T @ F |  2 * G.T @ G ] =
        #           =    2 * [F  -G].T @ [F  -G]
        def _fun(inp):
            alp, bet = inp[:degree_a], inp[degree_a:]
            diff = f @ alp - g @ bet
            obj_func = diff @ diff
            obj_grad = 2 * fg.T @ diff
            return obj_func, obj_grad

        # if no guess is provided, set the initial point as [ 1 / std(F @ 1) | 1 / std(G @ 1) ] then solve
        a0 = np.ones(degree_a) / np.sqrt(f.sum(axis=1).var(ddof=0) + self.eps) if a0 is None else a0
        b0 = np.ones(degree_b) / np.sqrt(g.sum(axis=1).var(ddof=0) + self.eps) if b0 is None else b0
        x0 = np.concatenate((a0, b0))
        s = minimize(
            _fun,
            jac=True,
            hess=lambda *_: 2 * fg.T @ fg,
            x0=x0,
            constraints=constraints,
            method=self.method,
            tol=self.tol,
            options={'maxiter': self.maxiter}
        )
        return s.x[:degree_a], s.x[degree_a:]

    @abstractmethod
    def _indicator(self, f, g, a0: Optional, b0: Optional) -> Tuple[Any, np.ndarray, np.ndarray]:
        pass


class DoubleKernelMetric(KernelBasedMetric, ABC):
    """Kernel-based metric computed using two different explicit kernels for the variables."""

    def __init__(self,
                 backend: Union[str, Backend] = 'numpy',
                 kernel_a: Union[int, Callable[[Any], list]] = 3,
                 kernel_b: Union[int, Callable[[Any], list]] = 3,
                 method: str = 'trust-constr',
                 maxiter: int = 1000,
                 eps: float = 1e-9,
                 tol: float = 1e-2,
                 use_lstsq: bool = True,
                 delta_independent: Optional[float] = None):
        """
        :param backend:
            The backend to use to compute the metric, or its alias.

        :param kernel_a:
            Either a callable f(a) yielding a list of variable's kernels, or an integer degree for a polynomial kernel.

        :param kernel_b:
            Either a callable g(b) yielding a list of variable's kernels, or an integer degree for a polynomial kernel.

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
        super(DoubleKernelMetric, self).__init__(
            backend=backend,
            method=method,
            maxiter=maxiter,
            eps=eps,
            tol=tol,
            use_lstsq=use_lstsq,
            delta_independent=delta_independent
        )

        # handle kernels
        if isinstance(kernel_a, int):
            degree_a = kernel_a
            kernel_a = lambda a: [a ** d for d in np.arange(degree_a) + 1]
        if isinstance(kernel_b, int):
            degree_b = kernel_b
            kernel_b = lambda b: [b ** d for d in np.arange(degree_b) + 1]
        self._kernel_a: Callable[[Any], list] = kernel_a
        self._kernel_b: Callable[[Any], list] = kernel_b

    def kernel_a(self, a) -> list:
        return self._kernel_a(a)

    def kernel_b(self, b) -> list:
        return self._kernel_b(b)

    def _compute(self, a, b) -> KernelBasedMetric.Result:
        # noinspection PyUnresolvedReferences
        a0, b0 = (None, None) if self.last_result is None else (self.last_result.alpha, self.last_result.beta)
        return self._result(a=a, b=b, kernel_a=True, kernel_b=True, a0=a0, b0=b0)


class SingleKernelMetric(KernelBasedMetric, ABC):
    """Kernel-based metric computed using a single kernel for the variables, then taking the maximal value."""

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
            Either a callable k(x) yielding a list of variable's kernels, or an integer degree for a polynomial kernel.

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
        super(SingleKernelMetric, self).__init__(
            backend=backend,
            method=method,
            maxiter=maxiter,
            eps=eps,
            tol=tol,
            use_lstsq=use_lstsq,
            delta_independent=delta_independent
        )

        # handle kernel
        if isinstance(kernel, int):
            degree = kernel
            kernel = lambda x: [x ** d for d in np.arange(degree) + 1]
        self._kernel: Callable[[Any], list] = kernel

    def kernel(self, x) -> list:
        """The list of kernels for the variables."""
        return self._kernel(x)

    def kernel_a(self, a) -> list:
        return self.kernel(a)

    def kernel_b(self, b) -> list:
        return self.kernel(b)

    def _compute(self, a, b) -> KernelBasedMetric.Result:
        # noinspection PyUnresolvedReferences
        a0, b0 = (None, None) if self.last_result is None else (self.last_result.alpha, self.last_result.beta)
        res_a = self._result(a=a, b=b, kernel_a=True, kernel_b=False, a0=a0, b0=None)
        res_b = self._result(a=a, b=b, kernel_a=False, kernel_b=True, a0=None, b0=b0)
        val_a = res_a.value
        val_b = res_b.value
        if val_a > val_b:
            value = val_a
            alpha = res_a.alpha
            degree = len(alpha)
            beta = np.concatenate((res_a.beta, np.zeros(degree - 1)))
        else:
            value = val_b
            beta = res_b.beta
            degree = len(beta)
            alpha = np.concatenate((res_b.alpha, np.zeros(degree - 1)))
        return KernelBasedMetric.Result(
            a=a,
            b=b,
            value=value,
            num_call=self.num_calls,
            metric=self,
            alpha=alpha,
            beta=beta,
        )
