import unittest
from abc import abstractmethod
from typing import Any, Iterable, Union, Type, Tuple

import numpy as np
import pytest

from maxcorr.backends import Backend


class TestBackend(unittest.TestCase):
    NUM: int = 5

    LENGTH: int = 10

    # noinspection PyTypeChecker
    @property
    @abstractmethod
    def backend(self) -> Backend:
        pytest.skip(reason="Abstract Test Class")

    # noinspection PyTypeChecker
    @property
    @abstractmethod
    def type(self) -> Type:
        pytest.skip(reason="Abstract Test Class")

    @abstractmethod
    def cast(self, v: list) -> Any:
        pytest.skip(reason="Abstract Test Class")

    def vectors(self,
                shape: Union[int, Iterable[int]] = 10,
                seed: int = 0,
                integer: bool = False,
                positive: bool = False) -> Tuple[list, Any]:
        rng = np.random.default_rng(seed=seed)
        vector = rng.integers(0, 10, size=shape) if integer else rng.normal(size=shape)
        vector = np.abs(vector) if positive else vector
        return vector.tolist(), self.cast(vector)

    def test_type(self) -> None:
        self.assertEqual(self.backend.type, self.type, msg="Type property should match expected type")

    def test_comply(self) -> None:
        ref, vec = self.vectors()
        self.assertTrue(self.backend.comply(v=vec), msg="Comply method should return True for compliant vector")
        self.assertFalse(self.backend.comply(v=ref), msg="Comply method should return False for non compliant vector")

    def test_floating(self) -> None:
        _, flt = self.vectors(integer=False)
        _, itg = self.vectors(integer=True)
        self.assertTrue(self.backend.floating(v=flt), msg="Floating method should return True for floating vector")
        self.assertFalse(self.backend.floating(v=itg), msg="Floating method should return False for integer vector")

    def test_cast(self) -> None:
        ref, _ = self.vectors()
        vec = self.backend.cast(v=ref)
        self.assertIsInstance(vec, self.type, msg="Cast method should return vector of correct type")
        self.assertTrue(np.allclose(ref, self.backend.numpy(vec)), msg="Cast method should return correct values")

    def test_item(self) -> None:
        # test correct functioning when single value is passed
        ref, vec = self.vectors(shape=1)
        ref_item, vec_item = float(ref[0]), self.backend.item(v=vec)
        self.assertIsInstance(vec_item, float, msg="Item should be of type float")
        self.assertEqual(vec_item, ref_item, msg="Item method should return correct item")
        # test exception raised when multiple values are passed
        _, vec = self.vectors(shape=self.LENGTH)
        # noinspection PyBroadException
        try:
            self.backend.item(v=vec)
        except Exception:
            pass

    def test_numpy(self) -> None:
        ref, vec = self.vectors()
        vec = self.backend.numpy(v=vec)
        self.assertIsInstance(vec, np.ndarray, msg="Numpy method should return a numpy vector")
        self.assertEqual(vec.tolist(), ref, msg="Numpy method should return correct values")

    def test_list(self) -> None:
        ref, vec = self.vectors()
        vec = self.backend.list(v=vec)
        self.assertIsInstance(vec, list, msg="List method should return a list")
        self.assertEqual(vec, ref, msg="List method should return correct values")

    def test_vector(self) -> None:
        ref, _ = self.vectors()
        vec = self.backend.vector(v=ref)
        self.assertIsInstance(vec, self.type, msg="Vector method should return vector of correct type")
        self.assertTrue(np.allclose(ref, self.backend.numpy(vec)), msg="Vector method should return correct values")

    def test_zeros(self) -> None:
        vec = self.backend.zeros(length=self.LENGTH)
        self.assertIsInstance(vec, self.type, msg="Zeros method should return vector of expected type")
        self.assertEqual(self.backend.list(vec), [0] * self.LENGTH, msg="Zeros method should return a vector of zeros")

    def test_ones(self) -> None:
        vec = self.backend.ones(length=self.LENGTH)
        self.assertIsInstance(vec, self.type, msg="Ones method should return vector of expected type")
        self.assertEqual(self.backend.list(vec), [1.0] * self.LENGTH, msg="Ones method should return a vector of ones")

    def test_dtype(self) -> None:
        _, vec = self.vectors()
        self.assertEqual(self.backend.dtype(vec), vec.dtype, msg="Dtype method should return the correct dtype")

    def test_squeeze(self) -> None:
        # to squeeze
        ref, vec = self.vectors(shape=(10, 1))
        ref = np.squeeze(ref).tolist()
        vec = self.backend.squeeze(vec)
        self.assertEqual(self.backend.shape(vec), (10,), msg="Squeeze method should return the correct shape")
        self.assertEqual(self.backend.list(vec), ref, msg="Squeeze method should return correct values")
        # not to squeeze
        ref, vec = self.vectors(shape=(5, 2))
        ref = np.squeeze(ref).tolist()
        vec = self.backend.squeeze(vec)
        self.assertEqual(self.backend.shape(vec), (5, 2), msg="Squeeze method should return the correct shape")
        self.assertEqual(self.backend.list(vec), ref, msg="Squeeze method should return correct values")

    def test_reshape(self) -> None:
        ref, vec = self.vectors(shape=10)
        ref = np.reshape(ref, (5, 2)).tolist()
        vec = self.backend.reshape(vec, shape=(-1, 2))
        self.assertEqual(self.backend.shape(vec), (5, 2), msg="Reshape method should return the correct shape")
        self.assertEqual(self.backend.list(vec), ref, msg="Reshape method should return correct values")

    def test_shape(self) -> None:
        ref, vec = self.vectors(shape=10)
        self.assertEqual(self.backend.shape(vec), np.shape(ref), msg="Shape method should return the correct shape")
        ref, vec = self.vectors(shape=(5, 2))
        self.assertEqual(self.backend.shape(vec), np.shape(ref), msg="Shape method should return the correct shape")

    def test_ndim(self) -> None:
        ref, vec = self.vectors()
        self.assertEqual(self.backend.ndim(vec), np.ndim(ref), msg="Ndim method should return the correct ndim")

    def test_len(self) -> None:
        ref, vec = self.vectors()
        self.assertEqual(self.backend.len(vec), len(ref), msg="Len method should return the correct length")

    def test_stack(self) -> None:
        refs = [self.vectors(seed=i)[0] for i in range(self.NUM)]
        vecs = [self.vectors(seed=i)[1] for i in range(self.NUM)]
        vecs = self.backend.stack(v=vecs, axis=1)
        self.assertEqual(
            self.backend.shape(vecs),
            (self.LENGTH, self.NUM),
            msg="Stacked vector should have the correct shape"
        )
        for i in range(self.NUM):
            self.assertEqual(self.backend.list(vecs[:, i]), refs[i], msg="Stack method should return original values")

    def test_abs(self) -> None:
        ref, vec = self.vectors()
        ref = np.abs(ref)
        vec = self.backend.numpy(self.backend.abs(v=vec))
        self.assertTrue(np.all(ref == vec), msg="Abs method should return the absolute values")

    def test_square(self) -> None:
        ref, vec = self.vectors()
        ref = np.square(ref)
        vec = self.backend.numpy(self.backend.square(v=vec))
        self.assertTrue(np.allclose(ref, vec), msg="Square method should return the squared values")

    def test_sqrt(self) -> None:
        ref, vec = self.vectors(positive=True)
        ref = np.sqrt(ref)
        vec = self.backend.numpy(self.backend.sqrt(v=vec))
        self.assertTrue(np.allclose(ref, vec), msg="Sqrt method should return the square root values")

    def test_matmul(self) -> None:
        ref1, vec1 = self.vectors(shape=(self.NUM, self.LENGTH), seed=0)
        ref2, vec2 = self.vectors(shape=self.LENGTH, seed=1)
        ref = np.matmul(ref1, ref2)
        vec = self.backend.numpy(self.backend.matmul(v=vec1, w=vec2))
        self.assertTrue(np.allclose(ref, vec), msg="Matmul method should return the matrix multiplication")

    def test_transpose(self) -> None:
        # transpose unidimensional
        ref, vec = self.vectors(shape=self.LENGTH, seed=0)
        ref = np.transpose(ref)
        vec = self.backend.numpy(self.backend.transpose(v=vec))
        self.assertTupleEqual(vec.shape, (1, self.LENGTH), msg="Transpose method should return correct shape")
        self.assertTrue(np.allclose(ref, vec), msg="Matmul method should return the matrix multiplication")
        # transpose bidimensional
        ref, vec = self.vectors(shape=(self.NUM, self.LENGTH), seed=0)
        ref = np.transpose(ref)
        vec = self.backend.numpy(self.backend.transpose(v=vec))
        self.assertTupleEqual(vec.shape, (self.LENGTH, self.NUM), msg="Transpose method should return correct shape")
        self.assertTrue(np.allclose(ref, vec), msg="Matmul method should return the matrix multiplication")

    def test_maximum(self) -> None:
        ref1, vec1 = self.vectors(seed=0)
        ref2, vec2 = self.vectors(seed=1)
        ref = np.maximum(ref1, ref2)
        vec = self.backend.numpy(self.backend.maximum(v=vec1, w=vec2))
        self.assertTrue(np.all(ref == vec), msg="Maximum method should return the element-wise maximum")

    def test_mean(self) -> None:
        ref, vec = self.vectors()
        ref = np.mean(ref)
        vec = self.backend.numpy(self.backend.mean(v=vec))
        self.assertTrue(np.isclose(ref, vec), msg="Mean method should return the mean of the vector")

    def test_sum(self) -> None:
        ref, vec = self.vectors()
        ref = np.sum(ref)
        vec = self.backend.numpy(self.backend.sum(v=vec))
        self.assertTrue(np.isclose(ref, vec), msg="Sum method should return the sum of the vector")

    def test_cov(self) -> None:
        ref1, vec1 = self.vectors(seed=0)
        ref2, vec2 = self.vectors(seed=1)
        # noinspection PyTypeChecker
        ref = np.cov([ref1, ref2], ddof=0)
        vec = self.backend.numpy(self.backend.cov(v=vec1, w=vec2))
        self.assertTrue(np.allclose(ref, vec), msg="Cov method should return the covariance matrix of the vectors")

    def test_var(self) -> None:
        ref, vec = self.vectors()
        ref = np.var(ref, ddof=0)
        vec = self.backend.numpy(self.backend.var(v=vec))
        self.assertTrue(np.isclose(ref, vec), msg="Var method should return the variance of the vector")

    def test_std(self) -> None:
        ref, vec = self.vectors()
        ref = np.std(ref, ddof=0)
        vec = self.backend.numpy(self.backend.std(v=vec))
        self.assertTrue(np.isclose(ref, vec), msg="Std method should return the standard deviation of the vector")

    def test_center(self) -> None:
        ref, vec = self.vectors()
        ref = np.array(ref) - np.mean(ref)
        vec = self.backend.numpy(self.backend.center(v=vec))
        self.assertTrue(np.allclose(ref, vec), msg="Center method should return the zero-centered vector")

    def test_standardize(self) -> None:
        ref, vec = self.vectors()
        ref = (np.array(ref) - np.mean(ref)) / np.std(ref, ddof=0)
        vec = self.backend.numpy(self.backend.standardize(v=vec))
        self.assertTrue(np.allclose(ref, vec), msg="Standardize method should return the standardized vector")

    def test_lstsq(self) -> None:
        ref1, vec1 = self.vectors(shape=(self.LENGTH, self.NUM), seed=0)
        ref2, vec2 = self.vectors(shape=self.LENGTH, seed=1)
        ref = np.linalg.lstsq(ref1, ref2, rcond=None)[0]
        vec = self.backend.numpy(self.backend.lstsq(A=vec1, b=vec2))
        self.assertTrue(np.allclose(ref, vec), msg="Lstsq method should return the least-square problem solution")
