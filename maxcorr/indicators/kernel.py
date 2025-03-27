"""
Implementations of the method from "Generalized Disparate Impact for Configurable Fairness Solutions in ML" by Luca
Giuliani, Eleonora Misino and Michele Lombardi. The code has been partially taken and reworked from the repository
containing the code of the paper: https://github.com/giuluck/GeneralizedDisparateImpact/tree/main.
"""

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from itertools import chain, combinations_with_replacement
from math import prod
from typing import Tuple, Optional, List, Any, Union, Callable, Dict

import numpy as np
import scipy
from scipy.optimize import NonlinearConstraint, minimize

from maxcorr.backends import Backend
from maxcorr.indicators.indicator import CopulaIndicator
from maxcorr.typing import BackendType, SemanticsType, AlgorithmType


class KernelBasedIndicator(CopulaIndicator):
    """Kernel-based indicator computed using user-defined kernels to approximate the copula transformations.

    The computation is native in any backend, therefore gradient information is always retrieved when possible.
    """

    @dataclass(frozen=True, init=True, repr=False, eq=False)
    class Result(CopulaIndicator.Result):
        """Data class representing the results of a KernelBasedIndicator computation."""

        alpha: List[float] = field()
        """The coefficient vector for the f copula transformation."""

        beta: List[float] = field()
        """The coefficient vector for the f copula transformation."""

    def __init__(self,
                 backend: Union[Backend, BackendType],
                 semantics: SemanticsType,
                 method: str,
                 maxiter: int,
                 eps: float,
                 tol: float,
                 use_lstsq: bool,
                 delta_independent: Optional[float]):
        """
        :param backend:
            The backend to use to compute the indicator, or its alias.

        :param semantics:
            The semantics of the indicator.

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
        super(KernelBasedIndicator, self).__init__(backend=backend, semantics=semantics, eps=eps)
        self._method: str = method
        self._maxiter: int = maxiter
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
    def last_result(self) -> Optional[Result]:
        # override method to change output type to KernelBasedIndicator.Result
        return super(KernelBasedIndicator, self).last_result

    def __call__(self, a, b) -> Result:
        # override method to change output type to KernelBasedIndicator.Result
        return super(KernelBasedIndicator, self).__call__(a, b)

    @property
    def alpha(self) -> List[float]:
        """The alpha vector computed in the last execution."""
        assert self.last_result is not None, "The indicator has not been computed yet, no transformation can be used."
        return self.last_result.alpha

    @property
    def beta(self) -> List[float]:
        """The beta vector computed in the last execution."""
        assert self.last_result is not None, "The indicator has not been computed yet, no transformation can be used."
        return self.last_result.beta

    @property
    def method(self) -> str:
        """The optimization method as in scipy.optimize.minimize, either 'trust-constr' or 'SLSQP'."""
        return self._method

    @property
    def maxiter(self) -> int:
        """The maximal number of iterations before stopping the optimization process as in scipy.optimize.minimize."""
        return self._maxiter

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

    def _f(self, a) -> Any:
        # cast vector to float if non-floating input type
        a = a if self.backend.floating(a) else self.backend.cast(a, dtype=float)
        kernel = self.backend.stack([self.backend.center(fi) for fi in self.kernel_a(a)], axis=1)
        alpha = self.backend.cast(self.last_result.alpha, dtype=self.backend.dtype(a))
        return self.backend.matmul(kernel, alpha)

    def _g(self, b) -> Any:
        # cast vector to float if non-floating input type
        b = b if self.backend.floating(b) else self.backend.cast(b, dtype=float)
        kernel = self.backend.stack([self.backend.center(gi) for gi in self.kernel_b(b)], axis=1)
        beta = self.backend.cast(self.last_result.beta, dtype=self.backend.dtype(b))
        return self.backend.matmul(kernel, beta)

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

    def _result(self,
                a,
                b,
                kernel_a: bool,
                kernel_b: bool,
                a0: Optional,
                b0: Optional) -> Tuple[Any, List[float], List[float]]:
        # build the kernel matrices, compute their original degrees, and get the linearly independent indices
        # if the kernels should not be used, create a list of input features by transposing the vector/matrix
        # and then taking each column as a single vector in the list in order to allow for multidimensional inputs
        n = self.backend.len(a)
        f = self.kernel_a(a) if kernel_a else [v for v in self.backend.transpose(a)]
        g = self.kernel_b(b) if kernel_b else [v for v in self.backend.transpose(b)]
        (f_slim, f_indices), (g_slim, g_indices) = self._indices(f=f, g=g)
        # compute the (centered) slim matrices and the respective degrees
        f_slim = self.backend.stack([self.backend.center(fi) for fi in f_slim], axis=1)
        g_slim = self.backend.stack([self.backend.center(gi) for gi in g_slim], axis=1)
        degree_a, degree_b = len(f_indices), len(g_indices)
        # compute the indicator value and the coefficients alpha and beta using the slim matrices
        # handle trivial or simpler cases:
        #  - if both degrees are 1 there is no additional computation involved
        #  - if one degree is 1, center/standardize that vector and compute the other's coefficients using lstsq
        #  - if no degree is 1, use the nonlinear lstsq optimization routine via scipy.optimize
        alpha = self.backend.ones(1, dtype=self.backend.dtype(f_slim))
        beta = self.backend.ones(1, dtype=self.backend.dtype(g_slim))
        alpha_numpy, beta_numpy = np.ones(1), np.ones(1)
        if degree_a == 1 and degree_b == 1:
            pass
        elif degree_a == 1 and self.use_lstsq:
            f_slim = self.backend.standardize(f_slim, eps=self.eps)
            beta = self.backend.lstsq(A=g_slim, b=self.backend.reshape(f_slim, shape=-1))
            beta_numpy = self.backend.numpy(beta)
        elif degree_b == 1 and self.use_lstsq:
            g_slim = self.backend.standardize(g_slim, eps=self.eps)
            alpha = self.backend.lstsq(A=f_slim, b=self.backend.reshape(g_slim, shape=-1))
            alpha_numpy = self.backend.numpy(alpha)
        else:
            f_numpy = self.backend.numpy(f_slim)
            g_numpy = self.backend.numpy(g_slim)
            fg_numpy = np.concatenate((f_numpy, -g_numpy), axis=1)

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
                diff_numpy = f_numpy @ alp - g_numpy @ bet
                obj_func = diff_numpy @ diff_numpy
                obj_grad = 2 * fg_numpy.T @ diff_numpy
                return obj_func, obj_grad

            # define the constraint
            #   - func:   var(G @ beta) --> = 1
            #   - grad: [ 0 | 2 * G.T @ G @ beta / n ]
            #   - hess: [ 0 |         0       ]
            #           [ 0 | 2 * G.T @ G / n ]
            cst_hess = np.zeros(shape=(degree_a + degree_b, degree_a + degree_b), dtype=float)
            cst_hess[degree_a:, degree_a:] = 2 * g_numpy.T @ g_numpy / n
            constraint = NonlinearConstraint(
                fun=lambda inp: np.var(g_numpy @ inp[degree_a:], ddof=0),
                jac=lambda inp: np.concatenate(([0] * degree_a, 2 * g_numpy.T @ g_numpy @ inp[degree_a:] / n)),
                hess=lambda *_: cst_hess,
                lb=1,
                ub=1
            )
            # if no guess is provided, set the initial point as [ 1 / std(F @ 1) | 1 / std(G @ 1) ] then solve
            if a0 is None:
                a0 = np.ones(degree_a) / np.sqrt(f_numpy.sum(axis=1).var(ddof=0) + self.eps)
            else:
                a0 = np.array(a0)[f_indices]
            if b0 is None:
                b0 = np.ones(degree_b) / np.sqrt(g_numpy.sum(axis=1).var(ddof=0) + self.eps)
            else:
                b0 = np.array(b0)[g_indices]
            x0 = np.concatenate((a0, b0))
            # noinspection PyTypeChecker
            s = minimize(
                _fun,
                jac=True,
                hess=lambda *_: 2 * fg_numpy.T @ fg_numpy,
                x0=x0,
                constraints=[constraint],
                method=self.method,
                tol=self.tol,
                options={'maxiter': self.maxiter}
            )
            alpha_numpy = s.x[:degree_a]
            beta_numpy = s.x[degree_a:]
            alpha = self.backend.cast(alpha_numpy, dtype=self.backend.dtype(f_slim))
            beta = self.backend.cast(beta_numpy, dtype=self.backend.dtype(g_slim))
        # compute the indicator value as the absolute value of the (mean) vector product
        fa = self.backend.standardize(self.backend.matmul(f_slim, alpha), eps=self.eps)
        gb = self.backend.standardize(self.backend.matmul(g_slim, beta), eps=self.eps)
        value = self.backend.mean(fa * gb)
        # reconstruct alpha and beta by adding zeros for the ignored indices, and normalize for ease of comparison
        alpha_full = np.zeros(len(f))
        alpha_full[f_indices] = alpha_numpy
        alpha_full = alpha_full / np.abs(alpha_full).sum()
        beta_full = np.zeros(len(g))
        beta_full[g_indices] = beta_numpy
        beta_full = beta_full / np.abs(beta_full).sum()
        # return the results, converting alpha and beta to lists of floats
        return value, [float(v) for v in alpha_full], [float(v) for v in beta_full]

    @staticmethod
    def _poly_kernel(v, degree: int, backend: Backend) -> list:
        # build the polynomial kernel expansion from the input vector <v>
        if backend.ndim(v) == 1:
            return [v ** d for d in np.arange(degree) + 1]
        else:
            _, features = backend.shape(v)
            iterables = [combinations_with_replacement(range(features), d + 1) for d in range(degree)]
            return [prod([v[:, i] for i in indices]) for indices in chain.from_iterable(iterables)]


class DoubleKernelIndicator(KernelBasedIndicator, ABC):
    """Kernel-based indicator computed using two different explicit kernels for the variables.

    The computation is native in any backend, therefore gradient information is always retrieved when possible.
    """

    algorithm: AlgorithmType = 'dk'

    def __init__(self,
                 kernel_a: Union[int, Callable[[Any], list]] = 3,
                 kernel_b: Union[int, Callable[[Any], list]] = 3,
                 backend: Union[Backend, BackendType] = 'numpy',
                 semantics: SemanticsType = 'hgr',
                 method: str = 'trust-constr',
                 maxiter: int = 1000,
                 eps: float = 1e-9,
                 tol: float = 1e-9,
                 use_lstsq: bool = True,
                 delta_independent: Optional[float] = None):
        """
        :param kernel_a:
            Either a callable f(a) yielding a list of variable's kernels, or an integer degree for a polynomial kernel.

        :param kernel_b:
            Either a callable g(b) yielding a list of variable's kernels, or an integer degree for a polynomial kernel.

        :param backend:
            The backend to use to compute the indicator, or its alias.

        :param semantics:
            The semantics of the indicator.

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
        super(DoubleKernelIndicator, self).__init__(
            backend=backend,
            semantics=semantics,
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
            kernel_a = lambda a: KernelBasedIndicator._poly_kernel(a, degree=degree_a, backend=self.backend)
        if isinstance(kernel_b, int):
            degree_b = kernel_b
            kernel_b = lambda b: KernelBasedIndicator._poly_kernel(b, degree=degree_b, backend=self.backend)
        self._kernel_a: Callable[[Any], list] = kernel_a
        self._kernel_b: Callable[[Any], list] = kernel_b

    def kernel_a(self, a) -> list:
        return self._kernel_a(a)

    def kernel_b(self, b) -> list:
        return self._kernel_b(b)

    def _compute(self, a, b) -> Tuple[Any, Dict[str, Any]]:
        # noinspection PyUnresolvedReferences
        a0, b0 = (None, None) if self.last_result is None else (self.last_result.alpha, self.last_result.beta)
        value, alpha, beta = self._result(a=a, b=b, kernel_a=True, kernel_b=True, a0=a0, b0=b0)
        return value, dict(alpha=alpha, beta=beta)


class SingleKernelIndicator(KernelBasedIndicator, ABC):
    """Kernel-based indicator computed using a single kernel for the variables, then taking the maximal value.

    The computation is native in any backend, therefore gradient information is always retrieved when possible.
    """

    algorithm: AlgorithmType = 'sk'

    def __init__(self,
                 kernel: Union[int, Callable[[Any], list]] = 3,
                 backend: Union[Backend, BackendType] = 'numpy',
                 semantics: SemanticsType = 'hgr',
                 method: str = 'trust-constr',
                 maxiter: int = 1000,
                 eps: float = 1e-9,
                 tol: float = 1e-9,
                 use_lstsq: bool = True,
                 delta_independent: Optional[float] = None):
        """
        :param kernel:
            Either a callable k(x) yielding a list of variable's kernels, or an integer degree for a polynomial kernel.

        :param backend:
            The backend to use to compute the indicator, or its alias.

        :param semantics:
            The semantics of the indicator.

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
        super(SingleKernelIndicator, self).__init__(
            backend=backend,
            semantics=semantics,
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
            kernel = lambda x: KernelBasedIndicator._poly_kernel(x, degree=degree, backend=self.backend)
        self._kernel: Callable[[Any], list] = kernel

    def kernel(self, x) -> list:
        """The list of kernels for the variables."""
        return self._kernel(x)

    def kernel_a(self, a) -> list:
        return self.kernel(a)

    def kernel_b(self, b) -> list:
        return self.kernel(b)

    def _compute(self, a, b) -> Tuple[Any, Dict[str, Any]]:
        # noinspection PyUnresolvedReferences
        a0, b0 = (None, None) if self.last_result is None else (self.last_result.alpha, self.last_result.beta)
        val_a, alpha_a, beta_a = self._result(a=a, b=b, kernel_a=True, kernel_b=False, a0=a0, b0=None)
        val_b, alpha_b, beta_b = self._result(a=a, b=b, kernel_a=False, kernel_b=True, a0=None, b0=b0)
        if val_a > val_b:
            value = val_a
            alpha = alpha_a
            padding = len(beta_b) - len(beta_a)
            beta = np.concatenate((beta_a, np.zeros(padding)))
        else:
            value = val_b
            beta = beta_b
            padding = len(alpha_a) - len(alpha_b)
            alpha = np.concatenate((alpha_b, np.zeros(padding)))
        return value, dict(alpha=alpha, beta=beta)
