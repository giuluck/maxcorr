"""
Implementation of the method from "Fairness-Aware Neural Renyi Minimization for Continuous Features" by Vincent Grari,
Sylvain Lamprier and Marcin Detyniecki. The code has been partially taken and reworked from the repository containing
the code of the paper: https://github.com/fairml-research/HGR_NN/tree/main.
"""
import importlib.util
from typing import Any, Iterable, Optional, Tuple, Callable, Union, Dict, List

from cfair.backends import Backend, NumpyBackend, TorchBackend, TensorflowBackend
from cfair.indicators.indicator import CopulaIndicator
from cfair.typing import BackendType, SemanticsType


class NeuralIndicator(CopulaIndicator):
    """Indicator computed using two neural networks to approximate the copula transformations.

    The computation is native in any backend, therefore gradient information is always retrieved when possible.
    """

    def __init__(self,
                 f_units: Optional[Iterable[int]] = (16, 16, 8),
                 g_units: Optional[Iterable[int]] = (16, 16, 8),
                 backend: Union[Backend, BackendType] = 'numpy',
                 semantics: SemanticsType = 'hgr',
                 num_features: Tuple[int, int] = (1, 1),
                 epochs_start: int = 1000,
                 epochs_successive: Optional[int] = 50,
                 eps: float = 1e-9):
        """
        :param f_units:
            The hidden units of the F copula network, or None for no F copula network.

        :param g_units:
            The hidden units of the G copula network, or None for no G copula network.

        :param backend:
            The backend to use to compute the indicator, or its alias.

        :param semantics:
            The semantics of the indicator.

        :param num_features:
            The number of features in the input data, to allow for multidimensional support.

        :param epochs_start:
            The number of training epochs in the first call.

        :param epochs_successive:
            The number of training epochs in the subsequent calls (fine-tuning of the pre-trained networks).

        :param eps:
            The epsilon value used to avoid division by zero in case of null standard deviation.
        """
        assert f_units is not None or g_units is not None, "Either f_units or g_units must not be None"
        super(NeuralIndicator, self).__init__(backend=backend, semantics=semantics, eps=eps)

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

        # build f transformation
        if f_units is None:
            assert num_features[0] == 1, "Transformation f is required since the <a> vector is multidimensional"
            f_net, f_opt = NeuralIndicator._DummyNetwork(), NeuralIndicator._DummyOptimizer()
        else:
            f_units = tuple(f_units)
            f_net, f_opt = build_fn(units=[num_features[0], *f_units])
        # build g transformation
        if g_units is None:
            assert num_features[1] == 1, "Transformation g is required since the <b> vector is multidimensional"
            g_net, g_opt = NeuralIndicator._DummyNetwork(), NeuralIndicator._DummyOptimizer()
        else:
            g_units = tuple(g_units)
            g_net, g_opt = build_fn(units=[num_features[1], *g_units])

        # store state
        self._unitsF, self._netF, self._optF = f_units, f_net, f_opt
        self._unitsG, self._netG, self._optG = g_units, g_net, g_opt
        self._num_features: Tuple[int, int] = num_features
        self._epochs_start: int = epochs_start
        self._epochs_successive: int = epochs_successive
        self._neural_backend: Backend = neural_backend
        self._train_fn: Callable[[Any, Any], None] = train_fn

    @property
    def f_units(self) -> Optional[Tuple[int]]:
        """The hidden units of the F copula network, or None if no F copula network."""
        return self._unitsF

    @property
    def g_units(self) -> Optional[Tuple[int]]:
        """The hidden units of the G copula network, or None if no G copula network."""
        return self._unitsG

    @property
    def num_features(self) -> Tuple[int, int]:
        """The number of features in the input data, to allow for multidimensional support."""
        return self._num_features

    @property
    def epochs_start(self) -> int:
        """The number of training epochs in the first call."""
        return self._epochs_start

    @property
    def epochs_successive(self) -> int:
        """The number of training epochs in the subsequent calls (fine-tuning of the pre-trained networks)."""
        return self._epochs_successive

    def _f(self, a) -> Any:
        n = self.backend.len(a)
        a = self._neural_backend.cast(a, dtype=float)
        a = self._neural_backend.reshape(a, shape=(n, self.num_features[0]))
        fa = self._netF(a)
        fa = self._neural_backend.reshape(fa, shape=n)
        return self._neural_backend.numpy(fa) if isinstance(self.backend, NumpyBackend) else fa

    def _g(self, b) -> Any:
        n = self.backend.len(b)
        b = self._neural_backend.cast(b, dtype=float)
        b = self._neural_backend.reshape(b, shape=(n, self.num_features[1]))
        gb = self._netG(b)
        gb = self._neural_backend.reshape(gb, shape=n)
        return self._neural_backend.numpy(gb) if isinstance(self.backend, NumpyBackend) else gb

    def _value(self, a, b) -> Tuple[Any, Dict[str, Any]]:
        # cast the vectors to the neural backend type
        n, (da, db) = self.backend.len(a), self.num_features
        a_cast = self._neural_backend.reshape(self._neural_backend.cast(a, dtype=float), shape=(n, da))
        b_cast = self._neural_backend.reshape(self._neural_backend.cast(b, dtype=float), shape=(n, db))
        for _ in range(self._epochs_start if self.num_calls == 0 else self._epochs_successive):
            self._train_fn(a_cast, b_cast)
        # compute the indicator value as the absolute value of the (mean) vector product
        # (since vectors are standardized) multiplied by the scaling factor
        value = self._hgr(a=a_cast, b=b_cast) * self._factor(a, b)
        value = self._neural_backend.item(value) if isinstance(self.backend, NumpyBackend) else value
        # return the result instance
        return value, dict()

    class _DummyNetwork:
        def __call__(self, x):
            return x

        @property
        def trainable_weights(self) -> list:
            return []

    class _DummyOptimizer:
        def zero_grad(self):
            return

        def step(self):
            return

        def apply_gradients(self, g):
            return

    def _hgr(self, a, b) -> Any:
        fa = self._neural_backend.standardize(self._netF(a), eps=self.eps)
        gb = self._neural_backend.standardize(self._netG(b), eps=self.eps)
        return self._neural_backend.mean(fa * gb)

    @staticmethod
    def _build_torch(units: List[int]) -> tuple:
        from torch.nn import Linear, Sequential, ReLU
        from torch.optim import Adam
        layers = []
        for inp, out in zip(units[:-1], units[1:]):
            layers += [Linear(inp, out), ReLU()]
        network = Sequential(*layers, Linear(units[-1], 1))
        optimizer = Adam(network.parameters(), lr=0.0005)
        return network, optimizer

    def _train_torch(self, a, b) -> None:
        self._optF.zero_grad()
        self._optG.zero_grad()
        loss = -self._hgr(a, b)
        loss.backward()
        self._optF.step()
        self._optG.step()

    @staticmethod
    def _build_tensorflow(units: List[int]) -> tuple:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.optimizers import Adam
        layers = [Dense(out, input_dim=inp, activation='relu') for inp, out in zip(units[:-1], units[1:])]
        network = Sequential([*layers, Dense(1, input_dim=units[-1])])
        optimizer = Adam(learning_rate=0.0005)
        return network, optimizer

    def _train_tensorflow(self, a, b) -> None:
        import tensorflow as tf
        with tf.GradientTape(persistent=True) as tape:
            loss = -self._hgr(a, b)
        f_grads = tape.gradient(loss, self._netF.trainable_weights)
        g_grads = tape.gradient(loss, self._netG.trainable_weights)
        self._optF.apply_gradients(zip(f_grads, self._netF.trainable_weights))
        self._optG.apply_gradients(zip(g_grads, self._netG.trainable_weights))
