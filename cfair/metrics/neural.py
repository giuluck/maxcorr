"""
HGR implementation of the method from "Fairness-Aware Neural Renyi Minimization for Continuous Features" by Vincent
Grari, Sylvain Lamprier and Marcin Detyniecki. The code has been partially taken and reworked from the repository
containing the code of the paper: https://github.com/fairml-research/HGR_NN/tree/main.
"""
import importlib.util
from typing import Any, Union, Iterable, Optional, Tuple, Callable

from cfair.backends import Backend, NumpyBackend, TorchBackend, TensorflowBackend
from cfair.metrics.metric import CopulaMetric


class NeuralHGR(CopulaMetric):
    """HGR indicator computed using two neural networks to approximate the copula transformations."""

    def __init__(self,
                 backend: Union[str, Backend] = 'numpy',
                 units: Iterable[int] = (16, 16, 8),
                 epochs_start: int = 1000,
                 epochs_successive: Optional[int] = 50,
                 eps: float = 1e-9):
        """
        :param backend:
            The backend to use to compute the metric, or its alias.

        :param units:
            The hidden units of the neural networks.

        :param epochs_start:
            The number of training epochs in the first call.

        :param epochs_successive:
            The number of training epochs in the subsequent calls (fine-tuning of the pre-trained networks).

        :param eps:
            The epsilon value used to avoid division by zero in case of null standard deviation.
        """
        super(NeuralHGR, self).__init__(backend=backend)
        # use default backend if it has a neural engine, otherwise prioritize torch and then tensorflow
        if isinstance(self.backend, TensorflowBackend):
            build_fn = self._build_tensorflow
            train_fn = self._train_tensorflow
            neural_backend = self.backend
        elif isinstance(self.backend, TorchBackend) or importlib.util.find_spec('torch') is not None:
            build_fn = self._build_torch
            train_fn = self._train_torch
            neural_backend = TorchBackend()
        elif importlib.util.find_spec('tensorflow') is not None:
            build_fn = self._build_tensorflow
            train_fn = self._train_tensorflow
            neural_backend = TensorflowBackend()
        elif isinstance(self.backend, NumpyBackend):
            raise ModuleNotFoundError(
                "NeuralHGR relies on neural networks and needs either pytorch or tensorflow installed even if "
                "NumpyBackend() is selected. Please install it via 'pip install torch' or 'pip install tensorflow'"
            )
        else:
            raise AssertionError(f"Unsupported backend f'{self.backend}")

        self._eps: float = eps
        self._units: Tuple[int] = tuple(units)
        self._epochs_start: int = epochs_start
        self._epochs_successive: int = epochs_successive
        self._neural_backend: Backend = neural_backend
        self._train_fn: Callable = train_fn
        self._netF, self._optF = build_fn()
        self._netG, self._optG = build_fn()

    @property
    def units(self) -> Tuple[int]:
        """The hidden units of the neural networks."""
        return self._units

    @property
    def epochs_start(self) -> int:
        """The number of training epochs in the first call."""
        return self._epochs_start

    @property
    def epochs_successive(self) -> int:
        """The number of training epochs in the subsequent calls (fine-tuning of the pre-trained networks)."""
        return self._epochs_successive

    @property
    def eps(self) -> float:
        """The epsilon value used to avoid division by zero in case of null standard deviation."""
        return self._eps

    def _indicator(self, a, b) -> Any:
        fa = self._neural_backend.standardize(self._netF(a), eps=self.eps)
        gb = self._neural_backend.standardize(self._netG(b), eps=self.eps)
        return self._neural_backend.mean(fa * gb)

    def _f(self, a) -> Any:
        a = self._neural_backend.cast(a, dtype=float)
        a = self._neural_backend.reshape(a, shape=(-1, 1))
        fa = self._netF(a)
        fa = self._neural_backend.reshape(fa, shape=-1)
        fa = self._neural_backend.standardize(fa, eps=self.eps)
        if self._backend is NumpyBackend():
            fa = self._neural_backend.numpy(fa)
        return fa

    def _g(self, b) -> Any:
        b = self._neural_backend.cast(b, dtype=float)
        b = self._neural_backend.reshape(b, shape=(-1, 1))
        gb = self._netG(b)
        gb = self._neural_backend.reshape(gb, shape=-1)
        gb = self._neural_backend.standardize(gb, eps=self.eps)
        if self._backend is NumpyBackend():
            gb = self._neural_backend.numpy(gb)
        return gb

    def _compute(self, a, b) -> CopulaMetric.Result:
        value = None
        # cast the vectors to the neural backend type
        a_cast = self._neural_backend.reshape(self._neural_backend.cast(a, dtype=float), shape=(-1, 1))
        b_cast = self._neural_backend.reshape(self._neural_backend.cast(b, dtype=float), shape=(-1, 1))
        for _ in range(self._epochs_start if self.num_calls == 0 else self._epochs_successive):
            value = self._train_fn(a_cast, b_cast)
        if self.backend is NumpyBackend():
            value = self._neural_backend.numpy(value).item()
        return NeuralHGR.Result(
            a=a,
            b=b,
            value=value,
            num_call=self.num_calls,
            metric=self
        )

    def _build_torch(self) -> tuple:
        import torch
        layers = []
        for inp, out in zip([1, *self.units], [*self.units, 1]):
            layers += [torch.nn.Linear(inp, out), torch.nn.ReLU()]
        network = torch.nn.Sequential(*layers[:-1])
        return network, torch.optim.Adam(network.parameters(), lr=0.0005)

    def _train_torch(self, a, b) -> Any:
        self._optF.zero_grad()
        self._optG.zero_grad()
        metric = self._indicator(a, b)
        loss = -metric
        loss.backward()
        self._optF.step()
        self._optG.step()
        return metric

    def _build_tensorflow(self) -> tuple:
        import tensorflow as tf
        layers = [tf.keras.layers.Dense(out, activation='relu') for out in self.units]
        return tf.keras.Sequential([*layers, tf.keras.layers.Dense(1)]), tf.keras.optimizers.Adam(learning_rate=0.0005)

    def _train_tensorflow(self, a, b) -> Any:
        import tensorflow as tf
        with tf.GradientTape(persistent=True) as tape:
            metric = self._indicator(a, b)
            loss = -metric
        f_grads = tape.gradient(loss, self._netF.trainable_weights)
        g_grads = tape.gradient(loss, self._netG.trainable_weights)
        self._optF.apply_gradients(zip(f_grads, self._netF.trainable_weights))
        self._optG.apply_gradients(zip(g_grads, self._netG.trainable_weights))
        return metric
