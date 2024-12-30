"""
Implementations of the method from "The Randomized Dependence Coefficient" by David Lopez-Paz, Philipp Hennig, and
Bernhard Schölkopf. The code has been partially taken and reworked from the repository of "Fairness-Aware Neural Rényi
Minimization for Continuous Features", which used it as a baseline: https://github.com/fairml-research/HGR_NN/.
"""

from typing import Dict, Any, Union, Callable, Tuple

import numpy as np
from scipy.stats import rankdata

from maxcorr.backends import Backend
from maxcorr.indicators.indicator import Indicator
from maxcorr.typing import BackendType, SemanticsType, AlgorithmType


class RandomizedIndicator(Indicator):
    """Indicator computed using the Randomized Dependence Coefficient (RDC).

    The computation relies on numpy, therefore no gradient information is returned for any backend.
    Moreover, this indicator supports univariate input data only.
    """

    algorithm: AlgorithmType = 'rdc'

    def __init__(self,
                 functions: Callable[[np.ndarray], np.ndarray] = np.sin,
                 backend: Union[Backend, BackendType] = 'numpy',
                 semantics: SemanticsType = 'hgr',
                 projections: int = 20,
                 scale: float = 1.0 / 6,
                 repetitions: int = 1):
        """
        :param functions:
            The class of functions used to build the random projections.

        :param backend:
            The backend to use to compute the indicator, or its alias.

        :param semantics:
            The semantics of the indicator.

        :param projections:
            The number of random projections to use.

        :param scale:
            The scale parameter.

        :param repetitions:
            The number of times to compute the RDC and return the median (for stability).
        """
        super(RandomizedIndicator, self).__init__(backend=backend, semantics=semantics)
        self._function: Callable[[np.ndarray], np.ndarray] = functions
        self._projections: int = projections
        self._scale: float = scale
        self._repetitions: int = repetitions

    @property
    def projections(self) -> int:
        """The number of random projections to use."""
        return self._projections

    @property
    def scale(self) -> float:
        """The scale parameter."""
        return self._scale

    @property
    def repetitions(self) -> int:
        """The number of times to compute the RDC and return the median (for stability)."""
        return self._repetitions

    def function(self, v: np.ndarray) -> np.ndarray:
        """The class of functions used to build the random projections.

        :param v:
            The input vector.

        :result:
            The mapped vector.
        """
        return self._function(v)

    def _compute(self, a, b) -> Tuple[Any, Dict[str, Any]]:
        a = self.backend.squeeze(a)
        b = self.backend.squeeze(b)
        if self.backend.ndim(a) != 1 or self.backend.ndim(b) != 1:
            raise ValueError('RandomizedIndicator can only handle one-dimensional vectors')
        value = RandomizedIndicator._rdc(
            x=self.backend.numpy(a),
            y=self.backend.numpy(b),
            f=self.function,
            k=self.projections,
            s=self.scale,
            n=self.repetitions
        )
        return self.backend.cast(value, dtype=self.backend.dtype(a)), dict()

    # noinspection PyPep8Naming
    @staticmethod
    def _rdc(x, y, f, k, s, n):
        if n > 1:
            values = []
            for i in range(n):
                try:
                    values.append(RandomizedIndicator._rdc(x, y, f, k, s, 1))
                except np.linalg.linalg.LinAlgError:
                    pass
            return np.median(values)

        if len(x.shape) == 1:
            x = x.reshape((-1, 1))
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        # Copula Transformation
        cx = np.column_stack([rankdata(xc, method='ordinal') for xc in x.T]) / float(x.size)
        cy = np.column_stack([rankdata(yc, method='ordinal') for yc in y.T]) / float(y.size)

        # Add a vector of ones so that w.x + b is just a dot product
        O = np.ones(cx.shape[0])
        X = np.column_stack([cx, O])
        Y = np.column_stack([cy, O])

        # Random linear projections
        Rx = (s / X.shape[1]) * np.random.randn(X.shape[1], k)
        Ry = (s / Y.shape[1]) * np.random.randn(Y.shape[1], k)
        X = np.dot(X, Rx)
        Y = np.dot(Y, Ry)

        # Apply non-linear function to random projections
        fX = f(X)
        fY = f(Y)

        # Compute full covariance matrix
        C = np.cov(np.hstack([fX, fY]).T)

        # Due to numerical issues, if k is too large,
        # then rank(fX) < k or rank(fY) < k, so we need
        # to find the largest k such that the eigenvalues
        # (canonical correlations) are real-valued
        k0 = k
        lb = 1
        ub = k
        while True:
            # Compute canonical correlations
            Cxx = C[:k, :k]
            Cyy = C[k0:k0 + k, k0:k0 + k]
            Cxy = C[:k, k0:k0 + k]
            Cyx = C[k0:k0 + k, :k]

            eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy), np.dot(np.linalg.pinv(Cyy), Cyx)))

            # Binary search if k is too large
            if not (np.all(np.isreal(eigs)) and
                    0 <= np.min(eigs) and
                    np.max(eigs) <= 1):
                ub -= 1
                k = (ub + lb) // 2
                continue
            if lb == ub:
                break
            lb = k
            if ub == lb + 1:
                k = ub
            else:
                k = (ub + lb) // 2

        return np.sqrt(np.max(eigs))
