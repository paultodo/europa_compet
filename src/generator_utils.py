import numpy as np


def sample_time_index(X, output_horizon,
                      min_sample_steps=12, max_sample_steps=24,
                      seed=1234):
    """Sample time indices as a random distance of each other"""
    random_steps = (np.random.RandomState(seed)
                    .uniform(min_sample_steps,
                             max_sample_steps,
                             size=X.shape[0])
                    .astype('int32')
                    .cumsum(axis=0))
    random_steps = random_steps[random_steps < len(X) - output_horizon]
    return random_steps


def create_2d_index(X, time_index):
    indices = []
    for time_step in time_index:
        for feature_no in np.arange(X.shape[1]):
            indices.append(np.array([time_step, feature_no]))
    return np.array(indices)


def get_batch_single_input(data_sources, index, hist_length, output_horizon,
                           dtype='float32'):
    x_b = np.zeros(shape=(len(index), hist_length, 1), dtype=dtype)
    y_b = np.zeros(shape=(len(index), output_horizon, 1), dtype=dtype)

    for ind, (time_step, feature_ind) in enumerate(index.tolist()):
        start_ = max(time_step - hist_length, 0)
        end_ = time_step + output_horizon
        data = data_sources['X'][start_:end_, feature_ind]
        if len(data) < hist_length + output_horizon:
            pad_length = hist_length + output_horizon - len(data)
            data = np.pad(data, pad_width=((pad_length, 0)),
                          mode='constant', constant_values=0)
        assert len(data) == hist_length + output_horizon
        x_b[ind, :, 0] = data[:hist_length]
        y_b[ind, :, 0] = data[-output_horizon:]
    return x_b, y_b, np.ones(shape=(len(index),))


def get_batch_all_inputs(data_sources, index, hist_length, output_horizon,
                         nb_inputs=1916, dtype='float32'):
    x_b = np.zeros(shape=(len(index), hist_length, nb_inputs), dtype=dtype)
    y_b = np.zeros(shape=(len(index), output_horizon, nb_inputs), dtype=dtype)

    assert len(index.shape) == 1 or index.shape[1] == 1
    index = index.ravel()
    for ind, time_step in enumerate(index.tolist()):
        start_ = max(time_step - hist_length, 0)
        end_ = time_step + output_horizon
        data = data_sources['X'][start_:end_, :]
        if len(data) < hist_length + output_horizon:
            pad_length = hist_length + output_horizon - len(data)
            data = np.pad(data, pad_width=((pad_length, 0), (0,0)),
                          mode='constant', constant_values=0)
        assert len(data) == hist_length + output_horizon
        x_b[ind, :, :] = data[:hist_length, :]
        y_b[ind, :, :] = data[-output_horizon:, :]
    return x_b, y_b, np.ones(shape=(len(index),))
