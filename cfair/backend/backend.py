from abc import abstractmethod
from typing import Tuple, final, Any, Type

import numpy as np


class Backend:
    """A stateless object representing a backend for vector operations. Apart from 'comply' and 'cast', all other
    functions expect inputs of a compliant type."""

    def __init__(self, backend) -> None:
        """
        :param backend:
            The module representing the backend to use.
        """
        self._backend = backend

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    @property
    def name(self) -> str:
        """An alias for the backend."""
        return self.__class__.__name__.replace('Backend', '').lower()

    @property
    @abstractmethod
    def type(self) -> Type:
        """The type of data handled by the backend."""
        pass

    @final
    def comply(self, v) -> bool:
        """Checks whether a vector complies with the backend (e.g., a numpy array for NumpyBackend).

        :param v:
            The input vector.

        :return:
            Whether the vector complies with the backend.
        """
        return isinstance(v, self.type)

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

    @final
    def list(self, v) -> list:
        """Casts the vector to a list.

        :param v:
            The input vector.

        :return:
            The cast vector.
        """
        return self.numpy(v).tolist()

    def zeros(self, length: int, dtype=None) -> Any:
        """Creates a vector of zeros with the given length and dtype.

        :param length:
            The length of the vector.

        :param dtype:
            The dtype of the vector.

        :return:
            The output vector.
        """
        return self._backend.zeros(length, dtype=dtype)

    def ones(self, length: int, dtype=None) -> Any:
        """Creates a vector of ones with the given length and dtype.

        :param length:
            The length of the vector.

        :param dtype:
            The dtype of the vector.

        :return:
            The output vector.
        """
        return self._backend.ones(length, dtype=dtype)

    # noinspection PyUnresolvedReferences, PyMethodMayBeStatic
    def dtype(self, v) -> Any:
        """Gets the type of the vector.

        :param v:
            The input vector.

        :return:
            The type of the vector.
        """
        return v.dtype

    # noinspection PyMethodMayBeStatic
    def shape(self, v) -> Tuple[int, ...]:
        """Gets the shape of the vector.

        :param v:
            The input vector.

        :return:
            A tuple representing the shape of the vector, along each dimension.
        """
        return tuple(v.shape)

    @final
    def ndim(self, v) -> int:
        """Gets the number of dimensions of the vector.

        :param v:
            The input vector.

        :return:
            The number of dimensions of the vector.
        """
        return len(self.shape(v))

    @final
    def len(self, v) -> int:
        """Gets the length of the vector on the first dimension.

        :param v:
            The input vector.

        :return:
            The length of the vector on the first dimension.
        """
        return self.shape(v)[0]

    @abstractmethod
    def stack(self, v: list) -> Any:
        """Stacks multiple vectors into a matrix.

        :param v:
            The list of vectors to stack.

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

    def matmul(self, v, w) -> Any:
        """Computes the matrix multiplication between two vectors/matrices.

        :param v:
            The first vector/matrix.

        :param w:
            The second vector/matrix.

        :return:
            The vector product <v, w>.
        """
        return self._backend.matmul(v, w)

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

    def mean(self, v) -> Any:
        """Computes the mean of the vector.

        :param v:
            The input vector.

        :return:
            The mean of the vector.
        """
        return self._backend.mean(v)

    @abstractmethod
    def var(self, v) -> Any:
        """Computes the variance of the vector.

        :param v:
            The input vector.

        :return:
            The variance of the vector.
        """
        pass

    @final
    def std(self, v) -> Any:
        """Computes the standard deviation of the vector.

        :param v:
            The input vector.

        :return:
            The standard deviation of the vector.
        """
        return self.sqrt(self.var(v))

    @final
    def standardize(self, v, eps: float = 1e-9) -> Any:
        """Standardizes a vector.

        :param v:
            The input vector.

        :param eps:
            The epsilon value used to avoid division by zero in case of null standard deviation.

        :return:
            The standardized vector.
        """
        return (v - self.mean(v)) / self.sqrt(self.var(v) + eps)

    @abstractmethod
    def lstsq(self, a, b) -> Any:
        """Runs least-square error fitting on the given vector and matrix.

        :param a:
            The lhs matrix A.

        :param b:
            The rhs vector b.

        :return:
            The optimal coefficient vector.
        """
        pass
