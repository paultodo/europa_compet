import numpy as np
import keras.backend as K
from src.layers import SliceTimeDimension


def test_slice_id():
    a = K.random_normal(shape=(32, 200, 100))
    slice_ = SliceTimeDimension(0, 200, 1)
    b = slice_(a)
    not_eq = K.not_equal(a, b)
    assert np.sum(K.eval(not_eq)) == 0


def test_slice_round():
    a = K.random_normal(shape=(32, 200, 100))
    slice_ = SliceTimeDimension(0, 200, 10)
    b = slice_(a)
    np.testing.assert_array_equal([d.value for d in b.shape], (32, 20, 100))
    for i in range(0, 200, 10):
        not_eq = K.not_equal(a[:, i, :], b[:, int(i / 10), :])
        assert np.sum(K.eval(not_eq)) == 0


def test_slice_not_round():
    a = K.random_normal(shape=(32, 200, 100))
    slice_ = SliceTimeDimension(0, 189, 10)
    b = slice_(a)
    np.testing.assert_array_equal([d.value for d in b.shape], (32, 19, 100))
    for i in range(0, 189, 10):
        not_eq = K.not_equal(a[:, i, :], b[:, int(i / 10), :])
        assert np.sum(K.eval(not_eq)) == 0


def test_slice_neg():
    a = K.random_normal(shape=(32, 200, 100))
    slice_ = SliceTimeDimension(0, -10, 10)
    b = slice_(a)
    np.testing.assert_array_equal([d.value for d in b.shape], (32, 19, 100))
    for i in range(0, 189, 10):
        not_eq = K.not_equal(a[:, i, :], b[:, int(i/10), :])
        assert np.sum(K.eval(not_eq)) == 0


def test_slice_step_neg():
    a = K.random_normal(shape=(32, 200, 100))
    slice_ = SliceTimeDimension(-1, -200, -10)
    b = slice_(a)
    np.testing.assert_array_equal([d.value for d in b.shape], (32, 20, 100))
    for i in range(-1, -200, -10):
        not_eq = K.not_equal(a[:, i, :], b[:, int(-(i+1)/10), :])
        assert np.sum(K.eval(not_eq)) == 0


if __name__ == '__main__':
    test_slice_id()
    test_slice_round()
    test_slice_not_round()
    test_slice_neg()
    test_slice_step_neg()
