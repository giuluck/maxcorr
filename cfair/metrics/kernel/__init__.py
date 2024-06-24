"""
GeDI and HGR implementations of the method from "Generalized Disparate Impact for Configurable Fairness Solutions in ML"
by Luca Giuliani, Eleonora Misino and Michele Lombardi, and "Enhancing the Applicability of Fair Learning with
Continuous Attributes" by Luca Giuliani and Michele Lombardi, respectively. The code has been partially taken and
reworked from the repositories containing the code of the paper, respectively:
- https://github.com/giuluck/GeneralizedDisparateImpact/tree/main
- https://github.com/giuluck/kernel-based-hgr/tree/main
"""

from cfair.metrics.kernel.abstract import KernelBasedMetric, DoubleKernelMetric, SingleKernelMetric
from cfair.metrics.kernel.gedi import KernelBasedGeDI, GeneralizedDisparateImpact
from cfair.metrics.kernel.hgr import KernelBasedHGR, DoubleKernelHGR, SingleKernelHGR
