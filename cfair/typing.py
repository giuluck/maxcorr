from typing import Literal

BackendType = Literal['numpy', 'tensorflow', 'torch']
"""The typeclass of the indicator backends."""

SemanticsType = Literal['hgr', 'gedi', 'nlc']
"""The typeclass of the indicator semantics."""

AlgorithmType = Literal['hgr', 'gedi', 'nlc']
"""The typeclass of the indicator algorithms."""
