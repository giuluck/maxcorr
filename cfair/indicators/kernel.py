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
from typing import Tuple, Optional, List, Any, Union, Callable

import numpy as np
import scipy
from scipy.optimize import NonlinearConstraint, minimize

from cfair.backends import Backend
from cfair.indicators.indicator import CopulaIndicator, HGRIndicator, GeDIIndicator, NLCIndicator


class KernelBasedIndicator(CopulaIndicator):
    """Kernel-based indicator computed using user-defined kernels to approximate the copula transformations."""

    @dataclass(frozen=True, init=True, repr=False, eq=False)
    class Result(CopulaIndicator.Result):
        """Data class representing the results of a KernelBasedIndicator computation."""

        alpha: List[float] = field()
        """The coefficient vector for the f copula transformation."""

        beta: List[float] = field()
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
            The backend to use to compute the indicator, or its alias.

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
        super(KernelBasedIndicator, self).__init__(backend=backend, eps=eps)
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
        kernel = self.backend.stack(self.kernel_a(a), axis=1)
        # noinspection PyUnresolvedReferences
        alpha = self.backend.cast(self.last_result.alpha)
        return self.backend.matmul(kernel, alpha)

    def _g(self, b) -> Any:
        kernel = self.backend.stack(self.kernel_b(b), axis=1)
        # noinspection PyUnresolvedReferences
        beta = self.backend.cast(self.last_result.beta)
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

    def _result(self, a, b, kernel_a: bool, kernel_b: bool, a0: Optional, b0: Optional) -> Result:
        # build the kernel matrices, compute their original degrees, and get the linearly independent indices
        f = self.kernel_a(a) if kernel_a else [a]
        g = self.kernel_b(b) if kernel_b else [b]
        (f_slim, f_indices), (g_slim, g_indices) = self._indices(f=f, g=g)
        # compute the slim matrices and the respective degrees
        f_slim = self.backend.stack(f_slim, axis=1)
        f_slim = f_slim - self.backend.mean(f_slim, axis=0)
        g_slim = self.backend.stack(g_slim, axis=1)
        g_slim = g_slim - self.backend.mean(g_slim, axis=0)
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
            n = len(a)
            f_numpy = self.backend.numpy(f_slim)
            g_numpy = self.backend.numpy(g_slim)
            fg_numpy = np.concatenate((f_slim, -g_slim), axis=1)

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
        # (since vectors are standardized) multiplied by the scaling factor
        fa = self.backend.standardize(self.backend.matmul(f_slim, alpha), eps=self.eps)
        gb = self.backend.standardize(self.backend.matmul(g_slim, beta), eps=self.eps)
        value = self.backend.mean(fa * gb) * self._factor(a=a, b=b)
        # reconstruct alpha and beta by adding zeros for the ignored indices, and normalize for ease of comparison
        alpha_full = np.zeros(len(f))
        alpha_full[f_indices] = alpha_numpy
        alpha_full = alpha_full / np.abs(alpha_full).sum()
        beta_full = np.zeros(len(g))
        beta_full[g_indices] = beta_numpy
        beta_full = beta_full / np.abs(beta_full).sum()
        # return the result instance, converting alpha and beta to lists of floats
        return KernelBasedIndicator.Result(
            a=a,
            b=b,
            value=value,
            num_call=self.num_calls,
            indicator=self,
            alpha=[float(v) for v in alpha_full],
            beta=[float(v) for v in beta_full],
        )


class DoubleKernelIndicator(KernelBasedIndicator, ABC):
    """Kernel-based indicator computed using two different explicit kernels for the variables."""

    def __init__(self,
                 kernel_a: Union[int, Callable[[Any], list]] = 3,
                 kernel_b: Union[int, Callable[[Any], list]] = 3,
                 backend: Union[str, Backend] = 'numpy',
                 method: str = 'trust-constr',
                 maxiter: int = 1000,
                 eps: float = 1e-9,
                 tol: float = 1e-9,
                 use_lstsq: bool = True,
                 delta_independent: Optional[float] = None):
        """
        :param backend:
            The backend to use to compute the indicator, or its alias.

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
        super(DoubleKernelIndicator, self).__init__(
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

    def _compute(self, a, b) -> KernelBasedIndicator.Result:
        # noinspection PyUnresolvedReferences
        a0, b0 = (None, None) if self.last_result is None else (self.last_result.alpha, self.last_result.beta)
        return self._result(a=a, b=b, kernel_a=True, kernel_b=True, a0=a0, b0=b0)


class SingleKernelIndicator(KernelBasedIndicator, ABC):
    """Kernel-based indicator computed using a single kernel for the variables, then taking the maximal value."""

    def __init__(self,
                 kernel: Union[int, Callable[[Any], list]] = 3,
                 backend: Union[str, Backend] = 'numpy',
                 method: str = 'trust-constr',
                 maxiter: int = 1000,
                 eps: float = 1e-9,
                 tol: float = 1e-9,
                 use_lstsq: bool = True,
                 delta_independent: Optional[float] = None):
        """
        :param backend:
            The backend to use to compute the indicator, or its alias.

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
        super(SingleKernelIndicator, self).__init__(
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

    def _compute(self, a, b) -> KernelBasedIndicator.Result:
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
        return KernelBasedIndicator.Result(
            a=a,
            b=b,
            value=value,
            num_call=self.num_calls,
            indicator=self,
            alpha=alpha,
            beta=beta,
        )


class DoubleKernelHGR(DoubleKernelIndicator, HGRIndicator):
    """Hirschfield-Gebelin-Renyi coefficient using two user-defined kernels as the copula transformations."""
    pass


class SingleKernelHGR(SingleKernelIndicator, HGRIndicator):
    """Hirschfield-Gebelin-Renyi coefficient using a single user-defined kernel as the copula transformation."""
    pass


class DoubleKernelGeDI(DoubleKernelIndicator, GeDIIndicator):
    """Generalized Disparate Impact using two user-defined kernels as the copula transformations."""

    # default value is to have the kernel on the first vector only
    def __init__(self,
                 kernel_a: Union[int, Callable[[Any], list]] = 3,
                 kernel_b: Union[int, Callable[[Any], list]] = 1,
                 backend: Union[str, Backend] = 'numpy',
                 method: str = 'trust-constr',
                 maxiter: int = 1000,
                 eps: float = 1e-9,
                 tol: float = 1e-9,
                 use_lstsq: bool = True,
                 delta_independent: Optional[float] = None):
        """
        :param backend:
            The backend to use to compute the indicator, or its alias.

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
        super(DoubleKernelGeDI, self).__init__(
            backend=backend,
            kernel_a=kernel_a,
            kernel_b=kernel_b,
            method=method,
            maxiter=maxiter,
            eps=eps,
            tol=tol,
            use_lstsq=use_lstsq,
            delta_independent=delta_independent
        )


class SingleKernelGeDI(SingleKernelIndicator, GeDIIndicator):
    """Generalized Disparate Impact using a single user-defined kernel as the copula transformation."""
    pass


class DoubleKernelNLC(DoubleKernelIndicator, NLCIndicator):
    """Non-Linear Covariance using two user-defined kernels as the copula transformations."""
    pass


class SingleKernelNLC(SingleKernelIndicator, NLCIndicator):
    """Non-Linear Covariance using a single user-defined kernel as the copula transformation."""
    pass
