from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Any, Union, Callable

import numpy as np
import scipy
from scipy.optimize import NonlinearConstraint, minimize

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
                 delta: float,
                 lasso: float):
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

        :param delta:
            A delta value used to decide whether two columns are linearly dependent.

        :param lasso:
            The amount of lasso regularization introduced when computing the metric (ignored for lstsq computation).
        """
        super(KernelBasedMetric, self).__init__(backend=backend)
        self._method: str = method
        self._maxiter: int = maxiter
        self._eps: float = eps
        self._tol: float = tol
        self._use_lstsq: bool = use_lstsq
        self._delta: float = delta
        self._lasso: lasso = lasso

    @abstractmethod
    def kernel_a(self, a) -> Any:
        """The kernel for the first variable."""
        pass

    @abstractmethod
    def kernel_b(self, b) -> Any:
        """The kernel for the second variable."""
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
        """The amount of lasso regularization introduced when computing the metric."""
        return self._lasso

    def _f(self, a) -> Any:
        kernel = self.kernel_a(a)
        # noinspection PyUnresolvedReferences
        alpha = self.backend.cast(self.last_result.alpha)
        fa = self.backend.matmul(kernel, alpha)
        return self.backend.standardize(fa)

    def _g(self, b) -> Any:
        kernel = self.kernel_b(b)
        # noinspection PyUnresolvedReferences
        beta = self.backend.cast(self.last_result.beta)
        gb = self.backend.matmul(kernel, beta)
        return self.backend.standardize(gb)

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
        """Computes the kernel coefficients for higher order kernels."""
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

    @abstractmethod
    def _indicator(self, a, b, kernel_a: bool, kernel_b: bool, a0: Optional, b0: Optional) -> Result:
        pass


class KernelBasedHGR(KernelBasedMetric, ABC):
    """Kernel-based metric interface where the computed indicator is HGR."""

    def _indicator(self, a, b, kernel_a: bool, kernel_b: bool, a0: Optional, b0: Optional) -> KernelBasedMetric.Result:
        backend = self.backend
        # build and center the kernel matrices
        f = self.kernel_a(a) if kernel_a else self.backend.reshape(a, shape=(-1, 1))
        f = f - self.backend.mean(f, axis=0)
        g = self.kernel_b(b) if kernel_b else self.backend.reshape(b, shape=(-1, 1))
        g = g - self.backend.mean(g, axis=0)
        # handle trivial or simpler cases:
        #  - if both degrees are 1, simply compute the projected vectors as standardized original vectors
        #  - if one degree is 1, standardize that vector and compute the other's coefficients using lstsq
        #  - if no degree is 1, use the optimization routine and compute the projected vectors from the coefficients
        alpha, beta = np.ones(1), np.ones(1)
        _, degree_a = self.backend.shape(f)
        _, degree_b = self.backend.shape(g)
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
        # return the HGR value as the absolute value of the (mean) vector product (since the vectors are standardized)
        value = backend.abs(backend.matmul(fa, gb) / backend.len(fa))
        return KernelBasedMetric.Result(
            a=a,
            b=b,
            value=value,
            num_call=self.num_calls,
            metric=self,
            alpha=alpha,
            beta=beta,
        )


class KernelBasedGeDI(KernelBasedMetric, ABC):
    """Kernel-based metric interface where the computed indicator is GeDI."""

    def _indicator(self, a, b, kernel_a: bool, kernel_b: bool, a0: Optional, b0: Optional) -> KernelBasedMetric.Result:
        raise NotImplementedError()


class DoubleKernelMetric(KernelBasedMetric, ABC):
    """Kernel-based metric computed using two different explicit kernels for the variables."""

    def __init__(self,
                 backend: Union[str, Backend] = 'numpy',
                 kernel_a: Union[int, Callable[[Any], Any]] = 3,
                 kernel_b: Union[int, Callable[[Any], Any]] = 3,
                 method: str = 'trust-constr',
                 maxiter: int = 1000,
                 eps: float = 1e-9,
                 tol: float = 1e-2,
                 use_lstsq: bool = True,
                 delta: float = 1e-2,
                 lasso: float = 0.0):
        """
        :param backend:
            The backend to use to compute the metric, or its alias.

        :param kernel_a:
            Either a callable f(a) representing the variable's kernel, or an integer degree for a polynomial kernel.

        :param kernel_b:
            Either a callable g(b) representing the variable's kernel, or an integer degree for a polynomial kernel.

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
            The amount of lasso regularization introduced when computing the metric (ignored for lstsq computation).
        """
        super(DoubleKernelMetric, self).__init__(
            backend=backend,
            method=method,
            maxiter=maxiter,
            eps=eps,
            tol=tol,
            use_lstsq=use_lstsq,
            delta=delta,
            lasso=lasso
        )

        # handle kernels
        if isinstance(kernel_a, int):
            degree_a = kernel_a
            kernel_a = lambda a: self.backend.stack([a ** d for d in np.arange(degree_a) + 1], axis=1)
        if isinstance(kernel_b, int):
            degree_b = kernel_b
            kernel_b = lambda b: self.backend.stack([b ** d for d in np.arange(degree_b) + 1], axis=1)
        self._kernel_a: Callable[[Any], Any] = kernel_a
        self._kernel_b: Callable[[Any], Any] = kernel_b

    def kernel_a(self, a) -> Any:
        return self._kernel_a(a)

    def kernel_b(self, b) -> Any:
        return self._kernel_b(b)

    def _compute(self, a, b) -> KernelBasedMetric.Result:
        # noinspection PyUnresolvedReferences
        a0, b0 = (None, None) if self.last_result is None else (self.last_result.alpha, self.last_result.beta)
        return self._indicator(a=a, b=b, kernel_a=True, kernel_b=True, a0=a0, b0=b0)


class SingleKernelMetric(KernelBasedMetric, ABC):
    """Kernel-based metric computed using a single kernel for the variables, then taking the maximal value."""

    def __init__(self,
                 backend: Union[str, Backend] = 'numpy',
                 kernel: Union[int, Callable[[Any], Any]] = 3,
                 method: str = 'trust-constr',
                 maxiter: int = 1000,
                 eps: float = 1e-9,
                 tol: float = 1e-2,
                 use_lstsq: bool = True,
                 delta: float = 1e-2,
                 lasso: float = 0.0):
        """
        :param backend:
            The backend to use to compute the metric, or its alias.

        :param kernel:
            Either a callable k(x) representing the variable's kernel, or an integer degree for a polynomial kernel.

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
            The amount of lasso regularization introduced when computing the metric (ignored for lstsq computation).
        """
        super(SingleKernelMetric, self).__init__(
            backend=backend,
            method=method,
            maxiter=maxiter,
            eps=eps,
            tol=tol,
            use_lstsq=use_lstsq,
            delta=delta,
            lasso=lasso
        )

        # handle kernel
        if isinstance(kernel, int):
            degree = kernel
            kernel = lambda x: self.backend.stack([x ** d for d in np.arange(degree) + 1], axis=1)
        self._kernel: Callable[[Any], Any] = kernel

    def kernel(self, x) -> Any:
        """The kernel for the variables."""
        return self._kernel(x)

    def kernel_a(self, a) -> Any:
        return self._kernel(a)

    def kernel_b(self, b) -> Any:
        return self._kernel(b)

    def _compute(self, a, b) -> KernelBasedMetric.Result:
        # noinspection PyUnresolvedReferences
        a0, b0 = (None, None) if self.last_result is None else (self.last_result.alpha, self.last_result.beta)
        res_a = self._indicator(a=a, b=b, kernel_a=True, kernel_b=False, a0=a0, b0=None)
        res_b = self._indicator(a=a, b=b, kernel_a=False, kernel_b=True, a0=None, b0=b0)
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


class DoubleKernelHGR(DoubleKernelMetric, KernelBasedHGR):
    """HGR indicator computed using two different explicit kernels for the variables."""
    pass


class SingleKernelHGR(SingleKernelMetric, KernelBasedHGR):
    """HGR indicator computed using a single kernel for the variables, then taking the maximal value."""
    pass


class DoubleKernelGeDI(DoubleKernelMetric, KernelBasedGeDI):
    """GeDI indicator computed using two different explicit kernels for the variables."""
    pass


class SingleKernelGeDI(SingleKernelMetric, KernelBasedGeDI):
    """GeDI indicator computed using a single kernel for the variables, then taking the maximal value."""
    pass


class GeneralizedDisparateImpact(DoubleKernelGeDI):
    """GeDI indicator computed as for its original formulation in the paper "Generalized Disparate Impact for
    Configurable Fairness Solutions in ML", i.e., using the kernel of the first vector only."""

    def __init__(self,
                 backend: Union[str, Backend] = 'numpy',
                 kernel: Union[int, Callable[[Any], Any]] = 3,
                 method: str = 'trust-constr',
                 maxiter: int = 1000,
                 eps: float = 1e-9,
                 tol: float = 1e-2,
                 use_lstsq: bool = True,
                 delta: float = 1e-2,
                 lasso: float = 0.0):
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

        :param delta:
            A delta value used to decide whether two columns are linearly dependent.

        :param lasso:
            The amount of lasso regularization introduced when computing the metric (ignored for lstsq computation).
        """
        super(DoubleKernelGeDI, self).__init__(
            backend=backend,
            kernel_a=kernel,
            kernel_b=1,
            method=method,
            maxiter=maxiter,
            eps=eps,
            tol=tol,
            use_lstsq=use_lstsq,
            delta=delta,
            lasso=lasso
        )
