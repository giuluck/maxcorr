"""
Implementations of the method from "Fairness-Aware Learning for Continuous Attributes and Treatments" by Jeremie Mary,
Clément Calauzènes, and Noureddine El Karoui. The code has been partially taken and reworked from the repository
containing the code of the paper: https://github.com/criteo-research/continuous-fairness/.
"""
import importlib.util
from abc import ABC
from math import pi, sqrt
from typing import Union, Any, Tuple, Dict

import numpy as np

from maxcorr.backends import Backend, TorchBackend
from maxcorr.indicators.indicator import Indicator
from maxcorr.typing import BackendType, SemanticsType, AlgorithmType


class DensityIndicator(Indicator, ABC):
    """Indicator computed using kernel density estimation techniques.

    The computation relies on torch, therefore a compatible version must be installed and no gradient information is
    returned if the chosen backend is Tensorflow. Moreover, this indicator supports univariate input data only.
    """

    algorithm: AlgorithmType = 'kde'

    def __init__(self,
                 backend: Union[Backend, BackendType] = 'numpy',
                 semantics: SemanticsType = 'hgr',
                 chi_square: bool = False,
                 damping: float = 1e-9):
        """
        :param backend:
            The backend to use to compute the indicator, or its alias.

        :param semantics:
            The semantics of the indicator.

        :param chi_square:
            Whether to use the chi square approximation or not.

        :param damping:
            The correction factor used in the computation of joint correlation.
        """
        super(DensityIndicator, self).__init__(backend=backend, semantics=semantics)
        if importlib.util.find_spec('torch') is None:
            raise ModuleNotFoundError("DensityIndicator relies on pytorch independently from any chosen backend. "
                                      "Please install it via 'pip install torch'.")
        self._chi_square: bool = chi_square
        self._damping: float = damping

    @property
    def chi_square(self) -> bool:
        """Whether to use the chi square approximation or not."""
        return self._chi_square

    @property
    def damping(self) -> float:
        """The correction factor used in the computation of joint correlation."""
        return self._damping

    def _compute(self, a, b) -> Tuple[Any, Dict[str, Any]]:
        import torch
        a = self.backend.squeeze(a)
        b = self.backend.squeeze(b)
        if self.backend.ndim(a) != 1 or self.backend.ndim(b) != 1:
            raise ValueError('DensityIndicator can only handle one-dimensional vectors')
        method = DensityIndicator.chi_2 if self.chi_square else DensityIndicator.hgr
        if isinstance(self.backend, TorchBackend):
            value = method(X=a, Y=b, density=DensityIndicator.kde, damping=self.damping)
        else:
            a = torch.tensor(self.backend.numpy(a), dtype=torch.float)
            b = torch.tensor(self.backend.numpy(b), dtype=torch.float)
            value = method(X=a, Y=b, density=DensityIndicator.kde, damping=self.damping)
            value = self.backend.cast(value.detach().cpu().numpy())
        return value, dict()

    # noinspection PyPep8Naming
    @staticmethod
    def _joint_2(X, Y, density, damping=1e-10):
        import torch
        X = (X - X.mean()) / X.std()
        Y = (Y - Y.mean()) / Y.std()
        data = torch.cat([X.unsqueeze(-1), Y.unsqueeze(-1)], -1)
        joint_density = density(data)

        nbins = int(min(50, 5. / joint_density.std))
        # nbins = np.sqrt( Y.size/5 )
        x_centers = torch.linspace(-2.5, 2.5, nbins)
        y_centers = torch.linspace(-2.5, 2.5, nbins)

        xx, yy = torch.meshgrid([x_centers, y_centers], indexing='ij')
        grid = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1)], -1)
        h2d = joint_density.pdf(grid) + damping
        h2d /= h2d.sum()
        return h2d

    # noinspection PyPep8Naming,PyIncorrectDocstring,DuplicatedCode
    @staticmethod
    def hgr(X, Y, density, damping=1e-10):
        """
        An estimator of the Hirschfeld-Gebelein-Rényi maximum correlation coefficient using Witsenhausen’s
        Characterization: HGR(x,y) is the second highest eigenvalue of the joint density on (x,y). We compute here the
        second eigenvalue on an empirical and discretized density estimated from the input data.
        :param X: A torch 1-D Tensor
        :param Y: A torch 1-D Tensor
        :param density: so far only kde is supported
        :return: numerical value between 0 and 1 (0: independent, 1:linked by a deterministic equation)
        """
        import torch
        h2d = DensityIndicator._joint_2(X, Y, density, damping=damping)
        marginal_x = h2d.sum(dim=1).unsqueeze(1)
        marginal_y = h2d.sum(dim=0).unsqueeze(0)
        Q = h2d / (torch.sqrt(marginal_x) * torch.sqrt(marginal_y))
        return torch.svd(Q)[1][1]

    # noinspection PyPep8Naming,PyIncorrectDocstring,PyRedundantParentheses,DuplicatedCode
    @staticmethod
    def chi_2(X, Y, density, damping=0):
        """
        The chi^2 divergence between the joint distribution on (x,y) and the product of marginals. This is know to be
        the square of an upper-bound on the Hirschfeld-Gebelein-Rényi maximum correlation coefficient. We compute it
        here on an empirical and discretized density estimated from the input data.
        :param X: A torch 1-D Tensor
        :param Y: A torch 1-D Tensor
        :param density: so far only kde is supported
        :return: numerical value between 0 and infinity (0: independent)
        """
        import torch
        h2d = DensityIndicator._joint_2(X, Y, density, damping=damping)
        marginal_x = h2d.sum(dim=1).unsqueeze(1)
        marginal_y = h2d.sum(dim=0).unsqueeze(0)
        Q = h2d / (torch.sqrt(marginal_x) * torch.sqrt(marginal_y))
        return ((Q ** 2).sum(dim=[0, 1]) - 1.)

    # noinspection PyPep8Naming
    @staticmethod
    def _joint_3(X, Y, Z, density, damping=1e-10):
        import torch
        X = (X - X.mean()) / X.std()
        Y = (Y - Y.mean()) / Y.std()
        Z = (Z - Z.mean()) / Z.std()
        data = torch.cat([X.unsqueeze(-1), Y.unsqueeze(-1), Z.unsqueeze(-1)], -1)
        joint_density = density(data)  # + damping

        nbins = int(min(50, 5. / joint_density.std))
        x_centers = torch.linspace(-2.5, 2.5, nbins)
        y_centers = torch.linspace(-2.5, 2.5, nbins)
        z_centers = torch.linspace(-2.5, 2.5, nbins)
        xx, yy, zz = torch.meshgrid([x_centers, y_centers, z_centers], indexing='ij')
        grid = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1), zz.unsqueeze(-1)], -1)

        h3d = joint_density.pdf(grid) + damping
        h3d /= h3d.sum()
        return h3d

    # noinspection PyPep8Naming,DuplicatedCode
    @staticmethod
    def hgr_cond(X, Y, Z, density):
        """
        An estimator of the function z -> HGR(x|z, y|z) where HGR is the Hirschfeld-Gebelein-Rényi maximum correlation
        coefficient computed using Witsenhausen’s Characterization: HGR(x,y) is the second highest eigenvalue of the
        joint density on (x,y). We compute here the second eigenvalue on an empirical and discretized density estimated
        from the input data.
        :param X: A torch 1-D Tensor
        :param Y: A torch 1-D Tensor
        :param Z: A torch 1-D Tensor
        :param density: so far only kde is supported
        :return: A torch 1-D Tensor of same size as Z. (0: independent, 1:linked by a deterministic equation)
        """
        import torch
        damping = 1e-10
        h3d = DensityIndicator._joint_3(X, Y, Z, density, damping=damping)
        marginal_xz = h3d.sum(dim=1).unsqueeze(1)
        marginal_yz = h3d.sum(dim=0).unsqueeze(0)
        Q = h3d / (torch.sqrt(marginal_xz) * torch.sqrt(marginal_yz))
        return np.array(([torch.svd(Q[:, :, i])[1][1] for i in range(Q.shape[2])]))

    # noinspection PyPep8Naming,PyRedundantParentheses,DuplicatedCode
    @staticmethod
    def chi_2_cond(X, Y, Z, density):
        """
        An estimator of the function z -> chi^2(x|z, y|z) where chi^2 is the chi^2 divergence between the joint
        distribution on (x,y) and the product of marginals. This is know to be the square of an upper-bound on the
        Hirschfeld-Gebelein-Rényi maximum correlation coefficient. We compute it here on an empirical and discretized
        density estimated from the input data.
        :param X: A torch 1-D Tensor
        :param Y: A torch 1-D Tensor
        :param Z: A torch 1-D Tensor
        :param density: so far only kde is supported
        :return: A torch 1-D Tensor of same size as Z. (0: independent)
        """
        import torch
        damping = 0
        h3d = DensityIndicator._joint_3(X, Y, Z, density, damping=damping)
        marginal_xz = h3d.sum(dim=1).unsqueeze(1)
        marginal_yz = h3d.sum(dim=0).unsqueeze(0)
        Q = h3d / (torch.sqrt(marginal_xz) * torch.sqrt(marginal_yz))
        return ((Q ** 2).sum(dim=[0, 1]) - 1.)

    # noinspection PyPep8Naming
    class kde:
        """
        A Gaussian KDE implemented in pytorch for the gradients to flow in pytorch optimization.

        Keep in mind that KDE are not scaling well with the number of dimensions and this implementation is not really
        optimized...
        """

        def __init__(self, x_train):
            n, d = x_train.shape

            self.n = n
            self.d = d

            self.bandwidth = (n * (d + 2) / 4.) ** (-1. / (d + 4))
            self.std = self.bandwidth

            self.train_x = x_train

        def pdf(self, x):
            import torch
            s = x.shape
            d = s[-1]
            s = s[:-1]
            assert d == self.d

            data = x.unsqueeze(-2)

            train_x = DensityIndicator._unsqueeze_multiple_times(self.train_x, 0, len(s))

            # noinspection PyTypeChecker
            pdf_values = (
                             torch.exp(-((data - train_x).norm(dim=-1) ** 2 / (self.bandwidth ** 2) / 2))
                         ).mean(dim=-1) / sqrt(2 * pi) / self.bandwidth

            return pdf_values

    # noinspection PyShadowingBuiltins
    @staticmethod
    def _unsqueeze_multiple_times(input, axis, times):
        """
        Utils function to unsqueeze tensor to avoid cumbersome code
        :param input: A pytorch Tensor of dimensions (D_1,..., D_k)
        :param axis: the axis to unsqueeze repeatedly
        :param times: the number of repetitions of the unsqueeze
        :return: the unsqueezed tensor. ex: dimensions (D_1,... D_i, 0,0,0, D_{i+1}, ... D_k) for unsqueezing 3x axis i.
        """
        output = input
        for i in range(times):
            output = output.unsqueeze(axis)
        return output
