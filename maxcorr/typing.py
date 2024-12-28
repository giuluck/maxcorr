from typing import Literal

BackendType = Literal['numpy', 'tensorflow', 'torch']
"""The typeclass of the indicator backends."""

SemanticsType = Literal['hgr', 'gedi', 'nlc']
"""The typeclass of the indicator semantics."""

AlgorithmType = Literal['dk', 'sk', 'nn', 'lat', 'kde', 'rdc']
"""The typeclass of the indicator algorithms."""
