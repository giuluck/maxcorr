from typing import Dict

from cfair.backend.backend import Backend
from cfair.backend.numpy import NumpyBackend
from cfair.backend.torch import TorchBackend

backends: Dict[str, Backend] = {
    'numpy': NumpyBackend(),
    'torch': TorchBackend()
}
