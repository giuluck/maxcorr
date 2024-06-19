from typing import Any, Type

import tensorflow as tf

from cfair.backend import Backend, TensorflowBackend
from test.backend.test_backend import TestBackend


class TestTorchBackend(TestBackend):

    @property
    def backend(self) -> Backend:
        return TensorflowBackend()

    @property
    def type(self) -> Type:
        return tf.Tensor

    def cast(self, v: list) -> Any:
        return tf.constant(v)
