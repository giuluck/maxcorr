from abc import abstractmethod
from typing import Tuple, Any, Type, Iterable, Union, Optional

import numpy as np

from maxcorr.typing import BackendType


class Backend:
    """A singleton object representing a backend for vector operations. Apart from 'comply' and 'cast', all other
    functions expect inputs of a compliant type."""

    _instance: Optional = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Backend, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, backend):
        """
        :param backend:
            The module representing the backend to use.
        """
        self._backend = backend

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def __eq__(self, other) -> bool:
        return self is other

    @property
    def name(self) -> BackendType:
        """An alias for the backend."""
        # noinspection PyTypeChecker
        return self.__class__.__name__.replace('Backend', '').lower()

    @property
    @abstractmethod
    def type(self) -> Type:
        """The type of data handled by the backend."""
        pass

    def comply(self, v) -> bool:
        """Checks whether a vector complies with the backend (e.g., a numpy array for NumpyBackend).

        :param v:
            The input vector.

        :return:
            Whether the vector complies with the backend.
        """
        return isinstance(v, self.type)

    @abstractmethod
    def floating(self, v) -> bool:
        """Checks whether a vector has floating point values.

        :param v:
            The input vector.

        :return:
            Whether the vector has floating point values.
        """
        pass

    @abstractmethod
    def cast(self, v, dtype=None) -> Any:
        """Casts the vector to the backend.

        :param v:
            The input vector.

        :param dtype:
            The dtype of the vector.

        :return:
            The cast vector.
        """
        pass

    @abstractmethod
    def item(self, v) -> float:
        """Returns the unique element of a vector as a floating point value.

        :param v:
            The input vector.

        :return:
            The unique element of the vector.
        """
        pass

    @abstractmethod
    def numpy(self, v, dtype=None) -> np.ndarray:
        """Casts the vector to a numpy vector.

        :param v:
            The input vector.

        :param dtype:
            The dtype of the vector.

        :return:
            The cast vector.
        """
        pass

    def list(self, v) -> list:
        """Casts the vector to a list.

        :param v:
            The input vector.

        :return:
            The cast vector.
        """
        return self.numpy(v).tolist()

    def vector(self, v: list, dtype=None) -> Any:
        """Creates a vector from an input list.

        :param v:
            The input values.

        :param dtype:
            The dtype of the vector.

        :return:
            The output vector.
        """
        return self.cast(v, dtype=dtype)

    def zeros(self, length: int, dtype=None) -> Any:
        """Creates a vector of zeros with the given length and dtype.

        :param length:
            The length of the vector.

        :param dtype:
            The dtype of the vector.

        :return:
            The output vector.
        """
        return self.vector([0] * length, dtype=dtype)

    def ones(self, length: int, dtype=None) -> Any:
        """Creates a vector of ones with the given length and dtype.

        :param length:
            The length of the vector.

        :param dtype:
            The dtype of the vector.

        :return:
            The output vector.
        """
        return self.vector([1] * length, dtype=dtype)

    # noinspection PyUnresolvedReferences, PyMethodMayBeStatic
    def dtype(self, v) -> Any:
        """Gets the type of the vector.

        :param v:
            The input vector.

        :return:
            The type of the vector.
        """
        return v.dtype

    def squeeze(self, v) -> Any:
        """Removes all the unitary dimensions of a multidimensional vector.

        :param v:
            The input multidimensional vector.

        :return:
            The output multidimensional vector.
        """
        return self._backend.squeeze(v)

    def reshape(self, v, shape: Union[int, Iterable[int]]) -> Any:
        """Reshapes the vector to the given shape.

        :param v:
            The input vector.

        :param shape:
            The expected shape of the output vector

        :return:
            The reshaped vector.
        """
        shape = (shape,) if isinstance(shape, int) else shape
        return self._backend.reshape(v, shape)

    # noinspection PyMethodMayBeStatic
    def shape(self, v) -> Tuple[int, ...]:
        """Gets the shape of the vector.

        :param v:
            The input vector.

        :return:
            A tuple representing the shape of the vector, along each dimension.
        """
        return tuple(v.shape)

    def ndim(self, v) -> int:
        """Gets the number of dimensions of the vector.

        :param v:
            The input vector.

        :return:
            The number of dimensions of the vector.
        """
        return len(self.shape(v))

    def len(self, v) -> int:
        """Gets the length of the vector on the first dimension.

        :param v:
            The input vector.

        :return:
            The length of the vector on the first dimension.
        """
        return self.shape(v)[0]

    @abstractmethod
    def stack(self, v: list, axis: Union[int, Iterable[int]] = 0) -> Any:
        """Stacks multiple vectors into a matrix.

        :param v:
            The list of vectors to stack.

        :param axis:
            The axis (dimensions) on which to stack the vectors.

        :return:
            The stacked matrix.
        """
        pass

    def abs(self, v) -> Any:
        """Computes the element-wise absolute values of the vector.

        :param v:
            The input vector.

        :return:
            The element-wise absolute values of the vector.
        """
        return self._backend.abs(v)

    def square(self, v) -> Any:
        """Computes the element-wise squares of the vector.

        :param v:
            The input vector.

        :return:
            The element-wise squares of the vector.
        """
        return self._backend.square(v)

    def sqrt(self, v) -> Any:
        """Computes the element-wise square root of the vector.

        :param v:
            The input vector.

        :return:
            The element-wise square root of the vector.
        """
        return self._backend.sqrt(v)

    @abstractmethod
    def matmul(self, v, w) -> Any:
        """Computes the matrix multiplication between two vectors/matrices.

        :param v:
            The first vector/matrix.

        :param w:
            The second vector/matrix.

        :return:
            The vector product <v, w>.
        """
        pass

    def transpose(self, v: Any) -> Any:
        """Transposes a vector/matrix.

        :param v:
            The input vector/matrix.

        :return:
            The transposed vector/matrix.
        """
        v = self.reshape(v, shape=(self.len(v), -1))
        return self._backend.transpose(v)

    def maximum(self, v, w) -> Any:
        """Computes the element-wise maximum between two vectors.

        :param v:
            The first vector.

        :param w:
            The second vector.

        :return:
            The element-wise maximum between the two vectors.
        """
        return self._backend.maximum(v, w)

    @abstractmethod
    def mean(self, v, axis: Union[None, int, Iterable[int]] = None) -> Any:
        """Computes the mean of the vector.

        :param v:
            The input vector.

        :param axis:
            The axis (dimensions) on which to compute the mean.

        :return:
            The mean of the vector.
        """
        pass

    @abstractmethod
    def sum(self, v, axis: Union[None, int, Iterable[int]] = None) -> Any:
        """Computes the sum of the vector.

        :param v:
            The input vector.

        :param axis:
            The axis (dimensions) on which to compute the sum.

        :return:
            The sum of the vector.
        """
        pass

    @abstractmethod
    def cov(self, v, w) -> Any:
        """Computes the covariance between the two vector.

        :param v:
            The first input vector.

        :param w:
            The second input vector.

        :return:
            The 2x2 covariance matrix between the two vectors.
        """
        pass

    @abstractmethod
    def var(self, v, axis: Union[None, int, Iterable[int]] = None) -> Any:
        """Computes the variance of the vector.

        :param v:
            The input vector.

        :param axis:
            The axis (dimensions) on which to compute the variance.

        :return:
            The variance of the vector.
        """
        pass

    def std(self, v, axis: Union[None, int, Iterable[int]] = None) -> Any:
        """Computes the standard deviation of the vector.

        :param v:
            The input vector.

        :param axis:
            The axis (dimension) on which to compute the standard deviation.

        :return:
            The standard deviation of the vector.
        """
        return self.sqrt(self.var(v, axis=axis))

    def center(self, v) -> Any:
        """Centers a vector on zero.

        :param v:
            The input vector.

        :return:
            The centered vector.
        """
        return v - self.mean(v)

    def standardize(self, v, eps: float = 1e-9) -> Any:
        """Standardizes a vector.

        :param v:
            The input vector.

        :param eps:
            The epsilon value used to avoid division by zero in case of null standard deviation.

        :return:
            The standardized vector.
        """
        return self.center(v) / self.sqrt(self.var(v) + eps)

    # noinspection PyPep8Naming
    @abstractmethod
    def lstsq(self, A, b) -> Any:
        """Runs least-square error fitting on the given vector and matrix.

        :param A:
            The lhs matrix A.

        :param b:
            The rhs vector b.

        :return:
            The optimal coefficient vector.
        """
        pass
