import numpy as np
from src.generator import EuropaGenerator


X = np.random.normal(size=(10000, 1916))

DEFAULT_PARAMS = dict(
    hist_length=12, output_horizon=12, batch_size=32,
    resample_time_index=False, time_step_per_epoch=100,
    time_index=None, inverse_value_aug=False, inverse_time_aug=False,
    all_inputs=False, shuffle=False, seed=1234
)


def test_dims_outputs():
    params = DEFAULT_PARAMS
    gen = EuropaGenerator(X=X, **params)
    x_b, y_b, _ = gen.next()
    shape_x = x_b.shape
    shape_y = y_b.shape

    assert shape_x[0] == params['batch_size']
    assert shape_x[1] == params['hist_length']
    assert shape_x[2] == 1

    assert shape_y[0] == params['batch_size']
    assert shape_y[1] == params['output_horizon']
    assert shape_y[2] == 1


def test_time_index_steps():
    params = DEFAULT_PARAMS
    gen = EuropaGenerator(X=X, **params)
    assert len(gen.time_index) == gen.time_step_per_epoch


def test_index_all_inputs():
    params = DEFAULT_PARAMS
    params['all_inputs'] = True
    gen = EuropaGenerator(X=X, **params)
    assert len(gen.index) == len(gen.time_index)


def test_index_one_input():
    params = DEFAULT_PARAMS
    params['all_inputs'] = False
    gen = EuropaGenerator(X=X, **params)
    assert len(gen.index) == len(gen.time_index) * gen.nb_total_inputs


def test_resample_index():
    params = DEFAULT_PARAMS
    params['resample_time_index'] = True
    gen = EuropaGenerator(X=X, **params)
    time_index_1 = gen.time_index
    gen.on_epoch_begin()
    time_index_2 = gen.time_index
    assert len(time_index_1) == len(time_index_2)
    assert np.any(time_index_1 != time_index_2)


if __name__ == '__main__':
    test_init()
    test_dims_outputs()
    test_time_index_steps()
    test_index_all_inputs()
    test_index_one_input()
    test_resample_index()
