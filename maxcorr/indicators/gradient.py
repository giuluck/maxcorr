"""
Implementation of the method from "Fairness-Aware Neural Rényi Minimization for Continuous Features" by Vincent Grari,
Sylvain Lamprier and Marcin Detyniecki. The code has been partially taken and reworked from the repository containing
the code of the paper: https://github.com/fairml-research/HGR_NN/tree/main.

It also includes a custom variant of gradient-based indicator using Tensorflow Lattice.
"""
import importlib.util
from abc import abstractmethod
from typing import Any, Iterable, Optional, Tuple, Callable, Union, Dict

from maxcorr.backends import Backend, NumpyBackend, TorchBackend, TensorflowBackend
from maxcorr.indicators.indicator import CopulaIndicator
from maxcorr.typing import BackendType, SemanticsType, AlgorithmType


class GradientIndicator(CopulaIndicator):
    """Indicator computed using two machine learning models trained using gradient-based computation."""

    def __init__(self,
                 backend: Union[Backend, BackendType],
                 semantics: SemanticsType,
                 f: Tuple[Any, Any, int],
                 g: Tuple[Any, Any, int],
                 train_fn: Callable[[Any, Any], None],
                 epochs_start: int,
                 epochs_successive: Optional[int],
                 eps: float):
        """
        :param backend:
            The backend to use to compute the indicator, or its alias.

        :param semantics:
            The semantics of the indicator.

        :param f:
            A tuple <model, optimizer, input_dimension> representing the F copula transformation.

        :param g:
            A tuple <model, optimizer, input_dimension> representing the G copula transformation.

        :param train_fn:
            A function that takes two vectors <a> and <b> as inputs and use them to train the models.

        :param epochs_start:
            The number of training epochs in the first call.

        :param epochs_successive:
            The number of training epochs in the subsequent calls (fine-tuning of the pre-trained networks).

        :param eps:
            The epsilon value used to avoid division by zero in case of null standard deviation.
        """
        super(GradientIndicator, self).__init__(backend=backend, semantics=semantics, eps=eps)
        self._modelF, self._optF, self._dimF = f
        self._modelG, self._optG, self._dimG = g
        self._epochs_start: int = epochs_start
        self._epochs_successive: int = epochs_successive
        self._train_fn: Callable[[Any, Any], None] = train_fn

    @property
    @abstractmethod
    def training_backend(self) -> Backend:
        """The internal backend used to train the models, which might differ from the user backend."""
        pass

    @property
    def epochs_start(self) -> int:
        """The number of training epochs in the first call."""
        return self._epochs_start

    @property
    def epochs_successive(self) -> int:
        """The number of training epochs in the subsequent calls (fine-tuning of the pre-trained networks)."""
        return self._epochs_successive

    @property
    def f_dimension(self) -> int:
        """The input dimension of the F copula transformation."""
        return self._dimF

    @property
    def g_dimension(self) -> int:
        """The input dimension of the G copula transformation."""
        return self._dimG

    def _f(self, a) -> Any:
        n = self.backend.len(a)
        a = self.training_backend.cast(a, dtype=float)
        a = self.training_backend.reshape(a, shape=(n, self.f_dimension))
        fa = self._modelF(a)
        fa = self.training_backend.reshape(fa, shape=n)
        return self.training_backend.numpy(fa) if isinstance(self.backend, NumpyBackend) else fa

    def _g(self, b) -> Any:
        n = self.backend.len(b)
        b = self.training_backend.cast(b, dtype=float)
        b = self.training_backend.reshape(b, shape=(n, self.g_dimension))
        gb = self._modelG(b)
        gb = self.training_backend.reshape(gb, shape=n)
        return self.training_backend.numpy(gb) if isinstance(self.backend, NumpyBackend) else gb

    def _compute(self, a, b) -> Tuple[Any, Dict[str, Any]]:
        # cast the vectors to the neural backend type
        n, da, db = self.backend.len(a), self.f_dimension, self.g_dimension
        a_cast = self.training_backend.reshape(self.training_backend.cast(a, dtype=float), shape=(n, da))
        b_cast = self.training_backend.reshape(self.training_backend.cast(b, dtype=float), shape=(n, db))
        for _ in range(self._epochs_start if self.num_calls == 0 else self._epochs_successive):
            self._train_fn(a_cast, b_cast)
        # compute the indicator value as the (mean) vector product
        value = self._hgr(a=a_cast, b=b_cast)
        # return the result instance cast to the correct type
        if not isinstance(self.backend, self.training_backend.__class__):
            value = self.backend.cast(self.training_backend.item(value), dtype=self.backend.dtype(a))
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
        fa = self.training_backend.standardize(self._modelF(a), eps=self.eps)
        gb = self.training_backend.standardize(self._modelG(b), eps=self.eps)
        return self.training_backend.mean(fa * gb)

    def _train_torch(self, a, b) -> None:
        self._optF.zero_grad()
        self._optG.zero_grad()
        try:
            loss = -self._hgr(a, b)
            loss.backward()
            self._optF.step()
            self._optG.step()
        except RuntimeError:
            pass

    def _train_tensorflow(self, a, b) -> None:
        import tensorflow as tf
        with tf.GradientTape(persistent=True) as tape:
            loss = -self._hgr(a, b)
        f_grads = tape.gradient(loss, self._modelF.trainable_weights)
        g_grads = tape.gradient(loss, self._modelG.trainable_weights)
        self._optF.apply_gradients(zip(f_grads, self._modelF.trainable_weights))
        self._optG.apply_gradients(zip(g_grads, self._modelG.trainable_weights))


class NeuralIndicator(GradientIndicator):
    """Indicator computed using two neural networks to approximate the copula transformations.

    The computation is native in any backend, therefore gradient information is always retrieved when possible.
    """

    algorithm: AlgorithmType = 'nn'

    def __init__(self,
                 f_units: Optional[Iterable[int]] = (16, 16, 8),
                 g_units: Optional[Iterable[int]] = (16, 16, 8),
                 backend: Union[Backend, BackendType] = 'numpy',
                 semantics: SemanticsType = 'hgr',
                 num_features: Tuple[int, int] = (1, 1),
                 epochs_start: int = 1000,
                 epochs_successive: Optional[int] = 50,
                 learning_rate: float = 0.0005,
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

        :param learning_rate:
            The learning rate of the Adam optimizer.

        :param eps:
            The epsilon value used to avoid division by zero in case of null standard deviation.
        """
        # use default backend if it has a neural engine, otherwise prioritize torch and then tensorflow
        if backend == 'tensorflow' or isinstance(backend, TensorflowBackend):
            build_fn = self._build_tensorflow
            train_fn = self._train_tensorflow
            training_backend = TensorflowBackend()
        elif backend == 'torch' or isinstance(backend, TorchBackend) or importlib.util.find_spec('torch') is not None:
            build_fn = self._build_torch
            train_fn = self._train_torch
            training_backend = TorchBackend()
        elif importlib.util.find_spec('tensorflow') is not None:
            build_fn = self._build_tensorflow
            train_fn = self._train_tensorflow
            training_backend = TensorflowBackend()
        elif backend == 'numpy' or isinstance(backend, NumpyBackend):
            raise ModuleNotFoundError(
                "NeuralIndicator relies on neural networks and needs either pytorch or tensorflow installed even if "
                "NumpyBackend() is selected. Please install it via 'pip install torch' or 'pip install tensorflow'"
            )
        else:
            raise AssertionError(f"Unsupported backend f'{self.backend}")

        super(NeuralIndicator, self).__init__(
            backend=backend,
            semantics=semantics,
            f=build_fn(units=f_units, dim=num_features[0], lr=learning_rate, name='f'),
            g=build_fn(units=g_units, dim=num_features[1], lr=learning_rate, name='g'),
            train_fn=train_fn,
            epochs_start=epochs_start,
            epochs_successive=epochs_successive,
            eps=eps,
        )
        self._unitsF: Optional[Tuple[int]] = None if f_units is None else tuple(f_units)
        self._unitsG: Optional[Tuple[int]] = None if g_units is None else tuple(g_units)
        self._learning_rate: float = learning_rate
        self._training_backend: Backend = training_backend

    @property
    def training_backend(self) -> Backend:
        return self._training_backend

    @property
    def f_units(self) -> Optional[Tuple[int]]:
        """The hidden units of the F copula network, or None if no F copula network."""
        return self._unitsF

    @property
    def g_units(self) -> Optional[Tuple[int]]:
        """The hidden units of the G copula network, or None if no G copula network."""
        return self._unitsG

    @property
    def learning_rate(self) -> float:
        """The learning rate of the Adam optimizer."""
        return self._learning_rate

    @staticmethod
    def _build_torch(units: Optional[Iterable[int]], dim: int, lr: float, name: str) -> Tuple[Any, Any, int]:
        import torch
        if units is None:
            assert dim == 1, f"Transformation {name} is required since its input vector is multidimensional"
            network = NeuralIndicator._DummyNetwork()
            optimizer = NeuralIndicator._DummyOptimizer()
        else:
            layers = []
            units = [dim, *units]
            for inp, out in zip(units[:-1], units[1:]):
                layers += [torch.nn.Linear(inp, out), torch.nn.ReLU()]
            network = torch.nn.Sequential(*layers, torch.nn.Linear(units[-1], 1))
            optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        return network, optimizer, dim

    @staticmethod
    def _build_tensorflow(units: Optional[Iterable[int]], dim: int, lr: float, name: str) -> Tuple[Any, Any, int]:
        import tensorflow as tf
        from tensorflow.keras.layers import Dense
        if units is None:
            assert dim == 1, f"Transformation {name} is required since its input vector is multidimensional"
            network = NeuralIndicator._DummyNetwork()
            optimizer = NeuralIndicator._DummyOptimizer()
        else:
            network = tf.keras.Sequential([Dense(out, activation='relu') for out in units] + [Dense(1)])
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        return network, optimizer, dim


class LatticeIndicator(GradientIndicator):
    """Indicator computed using two lattice models to approximate the copula transformations.

    The computation relies on tensorflow-lattice, therefore a compatible version must be installed and no gradient
    information is returned if the chosen backend is Torch.
    """

    algorithm: AlgorithmType = 'lat'

    def __init__(self,
                 f_sizes: Union[None, int, Iterable[int]] = (10,),
                 g_sizes: Union[None, int, Iterable[int]] = (10,),
                 backend: Union[Backend, BackendType] = 'numpy',
                 semantics: SemanticsType = 'hgr',
                 epochs_start: int = 1000,
                 epochs_successive: Optional[int] = 50,
                 learning_rate: float = 0.01,
                 eps: float = 1e-9,
                 f_kwargs: Optional[Dict[str, Any]] = None,
                 g_kwargs: Optional[Dict[str, Any]] = None):
        """
        :param f_sizes:
            The number of keypoints along each dimension of the F copula model, or None for no model.

        :param g_sizes:
            The number of keypoints along each dimension of the G copula model, or None for no model.

        :param backend:
            The backend to use to compute the indicator, or its alias.

        :param semantics:
            The semantics of the indicator.

        :param epochs_start:
            The number of training epochs in the first call.

        :param epochs_successive:
            The number of training epochs in the subsequent calls (fine-tuning of the pre-trained networks).

        :param eps:
            The epsilon value used to avoid division by zero in case of null standard deviation.

        :param f_kwargs:
            Additional arguments to be passed to the tfl.layers.Lattice instance representing the F copula model.

        :param g_kwargs:
            Additional arguments to be passed to the tfl.layers.Lattice instance representing the G copula model.
        """
        if importlib.util.find_spec('tensorflow') is None:
            raise ModuleNotFoundError("LatticeIndicator needs both tensorflow and tensorflow-lattice installed. "
                                      "Please install it via 'pip install tensorflow tensorflow-lattice'")

        if isinstance(f_sizes, int):
            f_sizes = (f_sizes,)
        elif f_sizes is not None:
            f_sizes = tuple(f_sizes)
        if isinstance(g_sizes, int):
            g_sizes = (g_sizes,)
        elif g_sizes is not None:
            g_sizes = tuple(g_sizes)

        super(LatticeIndicator, self).__init__(
            backend=backend,
            semantics=semantics,
            f=self._build_model(sizes=f_sizes, lr=learning_rate, kwargs=f_kwargs),
            g=self._build_model(sizes=g_sizes, lr=learning_rate, kwargs=g_kwargs),
            train_fn=self._train_tensorflow,
            epochs_start=epochs_start,
            epochs_successive=epochs_successive,
            eps=eps
        )

        self._sizesF: Optional[Tuple[int, ...]] = f_sizes
        self._sizesG: Optional[Tuple[int, ...]] = g_sizes
        self._learning_rate: float = learning_rate

    @property
    def training_backend(self) -> Backend:
        return TensorflowBackend()

    @property
    def f_sizes(self) -> Optional[Tuple[int, ...]]:
        """The number of keypoints along each dimension of the F copula model, or None for no model."""
        return self._sizesF

    @property
    def g_sizes(self) -> Optional[Tuple[int, ...]]:
        """The number of keypoints along each dimension of the G copula model, or None for no model."""
        return self._sizesG

    @property
    def learning_rate(self) -> float:
        """The learning rate of the Adam optimizer."""
        return self._learning_rate

    @staticmethod
    def _build_model(sizes: Optional[Iterable[int]],
                     lr: float,
                     kwargs: Optional[Dict[str, Any]]) -> Tuple[Any, Any, int]:
        import tensorflow as tf
        import tensorflow_lattice as tfl
        if sizes is None:
            dim = 1
            model = LatticeIndicator._DummyNetwork()
            optimizer = LatticeIndicator._DummyOptimizer()
        else:
            dim = len(list(sizes))
            kwargs = dict() if kwargs is None else kwargs
            model = tfl.layers.Lattice(lattice_sizes=sizes, units=1, **kwargs)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        return model, optimizer, dim
