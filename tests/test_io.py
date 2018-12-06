import numpy as np
import pytest
import os
import shutil
import h5py
from src.target_data_io import glob_adapt_files, glob_train_files, save_h5_input_file, \
    load_h5_input_file, load_full_cache_matrixes, \
    build_predict_matrix_from_cache_and_candidates_timestep, get_necessary_steps, fillna_t, \
    most_ancient_timestep_with_nan, most_recent_timestep_with_nan, first_non_nan_input
from start_prediction_server import load_prediction_matrix


def test_read_ndim_2(fake_input_dir):
    train_files = glob_train_files(fake_input_dir)
    adapt_files = glob_adapt_files(fake_input_dir, 10000)

    for data_file in adapt_files + train_files:
        X, t = load_h5_input_file(data_file)
        print(X.shape)
        # Corruption
        os.unlink(data_file)
        save_h5_input_file(data_file, X, t, output_dim = 2)
        X_copy, t_copy = load_h5_input_file(data_file)
        assert len(X_copy.shape) == 2


def test_noise_files(fake_input_dir):
    train_files = glob_train_files(fake_input_dir)
    adapt_files = glob_adapt_files(fake_input_dir, 10000)
    open(os.path.join(fake_input_dir, 'train/Xm1/', 'Xfake.h5'), 'w').close()
    open(os.path.join(fake_input_dir, 'train/', 'Xfake.h5'), 'w').close()
    open(os.path.join(fake_input_dir, 'adapt/', 'Xfake.h5'), 'w').close()
    new_train_files = glob_train_files(fake_input_dir)
    new_adapt_files = glob_adapt_files(fake_input_dir, 10000)
    assert len(train_files) == len(new_train_files)
    assert len(adapt_files) == len(new_adapt_files)


@pytest.mark.parametrize('eval_step', [0, 1, 10, 50])
def test_load_full_cache_matrixes(fake_input_dir, eval_step, hist_length=16*12*12):
    all_X, all_t = load_full_cache_matrixes(fake_input_dir, eval_step=0,
                                            steps_needed=hist_length)
    all_X = build_predict_matrix_from_cache_and_candidates_timestep(all_X, all_t, hist_length)
    assert all_X.shape[0] == hist_length


def test_t_empty_array(fake_input_dir, hist_length=16*12*12):
    data_file = os.path.join(fake_input_dir, 'adapt', 'X0.h5')
    X, t = load_h5_input_file(data_file)
    t = np.array([])
    os.unlink(data_file)
    save_h5_input_file(data_file, X, t, output_dim=2)
    new_X, new_t = load_h5_input_file(data_file)
    assert new_X.shape[0] == new_t.shape[0]
    np.testing.assert_array_equal(X, new_X)
    np.testing.assert_array_equal(new_t, np.array([np.nan] * X.shape[0]))



def test_t_diff_len_from_X(fake_input_dir, hist_length=16*12*12):
    data_file = os.path.join(fake_input_dir, 'adapt', 'X0.h5')
    X, t = load_h5_input_file(data_file)
    t = t[:-1]
    os.unlink(data_file)
    save_h5_input_file(data_file, X, t, output_dim=2)
    new_X, new_t = load_h5_input_file(data_file)
    assert new_X.shape[0] == new_t.shape[0]
    np.testing.assert_array_equal(X, new_X)
    np.testing.assert_array_equal(new_t, np.append(t, np.array([np.nan])))


def test_t_has_nan(fake_input_dir, hist_length=16*12*12):
    data_file = os.path.join(fake_input_dir, 'adapt', 'X0.h5')
    X, t = load_h5_input_file(data_file)

    corrupted_t = t.copy()
    corrupted_t[np.random.choice(len(t), len(t)//2, replace=False)] = np.nan
    corrupted_t[0] = np.nan
    corrupted_t[-1] = np.nan

    os.unlink(data_file)
    save_h5_input_file(data_file+'.corrupted', X, corrupted_t, output_dim=2)
    new_X, new_t = load_h5_input_file(data_file+'.corrupted')

    assert new_X.shape[0] == new_t.shape[0]
    np.testing.assert_array_equal(new_t, corrupted_t)
    np.testing.assert_array_equal(t, fillna_t(new_t))
    assert new_t.dtype == t.dtype
    np.testing.assert_array_equal(X, new_X)


def test_t_has_zeros(fake_input_dir, hist_length=16*12*12):
    data_file = os.path.join(fake_input_dir, 'adapt', 'X0.h5')
    X, t = load_h5_input_file(data_file)

    corrupted_t = t.copy()
    corrupted_t[np.random.choice(len(t), len(t)//2, replace=False)] = 0

    os.unlink(data_file)
    save_h5_input_file(data_file, X, corrupted_t, output_dim=2)
    new_X, new_t = load_h5_input_file(data_file)

    assert new_X.shape[0] == new_t.shape[0]
    np.testing.assert_array_equal(X, new_X)
    corrupted_t_equivalent = corrupted_t.copy()
    corrupted_t_equivalent[corrupted_t_equivalent == 0] = np.nan
    np.testing.assert_array_equal(new_t, corrupted_t_equivalent)
    np.testing.assert_array_equal(t, fillna_t(new_t))


"""
def test_data_shuffled(fake_input_dir, hist_length=16*12*12):
    data_file = os.path.join(fake_input_dir, 'adapt', 'X0.h5')
    X, t = load_h5_input_file(data_file)

    shuffle_indice = np.random.permutation(len(t))
    corrupted_t = t[shuffle_indice]
    corrupted_X = X[shuffle_indice]

    os.unlink(data_file)
    save_h5_input_file(data_file, corrupted_X, corrupted_t, output_dim=2)
    new_X, new_t = load_h5_input_file(data_file)

    assert new_X.shape[0] == new_t.shape[0]
    np.testing.assert_array_equal(X, new_X)
    np.testing.assert_array_equal(new_t, t)
"""


def test_X_ndim_4(fake_input_dir):
    data_file = os.path.join(fake_input_dir, 'adapt', 'X0.h5')
    X, t = load_h5_input_file(data_file)
    assert len(X.shape) == 2
    X = X[:, :, np.newaxis, np.newaxis]
    os.unlink(data_file)
    with h5py.File(data_file, driver = 'core', mode='w') as h5file:
        group = h5file
        group = group.create_group('X')
        group = group.create_group('value')
        group_X = group.create_group('X')
        group_X.create_dataset('value', data = X)

        group_t = group.create_group('t')
        group_t.create_dataset('value', data = t.astype(float).reshape(-1, 1))
    X, t = load_h5_input_file(data_file)
    assert len(X.shape) == 2


def test_X_ndim_5(fake_input_dir):
    data_file = os.path.join(fake_input_dir, 'adapt', 'X0.h5')
    X, t = load_h5_input_file(data_file)
    assert len(X.shape) == 2
    X = X[:, :, np.newaxis, np.newaxis, np.newaxis]
    os.unlink(data_file)
    with h5py.File(data_file, driver = 'core', mode='w') as h5file:
        group = h5file
        group = group.create_group('X')
        group = group.create_group('value')
        group_X = group.create_group('X')
        group_X.create_dataset('value', data = X)

        group_t = group.create_group('t')
        group_t.create_dataset('value', data = t.astype(float).reshape(-1, 1))
    X, t = load_h5_input_file(data_file)
    assert len(X.shape) == 2


def test_data_readonly(fake_input_dir):
    data_file = os.path.join(fake_input_dir, 'adapt', 'X0.h5')
    os.chmod(data_file, 0o400)
    X, t = load_h5_input_file(data_file)


def corrupt_t(t):
    corruption = np.random.choice(4)
    corrupted_t = t.copy()
    if corruption == 0:
        return np.array([])
    elif corruption == 1:
        return np.array([np.nan for _ in t])
    elif corruption == 2:
        corrupted_t = t.astype('float')
        corrupted_t[np.random.choice(len(t), len(t) // 2, replace=False)] = np.nan
        return corrupted_t
    elif corruption == 3:
        corrupted_t[np.random.choice(len(t), len(t) // 2, replace=False)] = 0
        return corrupted_t


def test_load_full_cache_matrixes_corrupted_t(fake_input_dir, tmpdir,
                                              hist_length=64*12*12):
    # Configuration
    test_input_dir = os.path.join(str(tmpdir), 'test_input')
    shutil.copytree(fake_input_dir, test_input_dir)

    # Transformation
    train_files = glob_train_files(test_input_dir)
    adapt_files = glob_adapt_files(test_input_dir, 10000)

    for data_file in adapt_files + train_files:
        X, t = load_h5_input_file(data_file)
        # Corruption
        t = corrupt_t(t)
        os.unlink(data_file)
        save_h5_input_file(data_file, X, t, output_dim = 2)

    # Evaluation
    for eval_step in range(-1, 60, 5):
        original_X, original_t = load_full_cache_matrixes(fake_input_dir, eval_step=eval_step,
                                                steps_needed=hist_length)
        X_with_corrupted_t, corrupted_t = load_full_cache_matrixes(test_input_dir, eval_step=eval_step,
                                                steps_needed=hist_length)
        assert len(corrupted_t) == len(original_t)
        np.testing.assert_array_equal(original_X, X_with_corrupted_t)
        np.testing.assert_array_equal(original_t[~np.isnan(corrupted_t)], corrupted_t[~np.isnan(corrupted_t)])
        np.testing.assert_array_equal(original_t, fillna_t(corrupted_t))


def load_data_matrix(input_path, eval_step, hist_length):
    all_X, all_t = load_full_cache_matrixes(input_path, eval_step=eval_step,
                                            steps_needed=hist_length)
    all_X = build_predict_matrix_from_cache_and_candidates_timestep(all_X, all_t,
                                                                    hist_length)
    assert all_X.shape[0] == hist_length
    return all_X


def test_load_predict_matrixes_corrupted_t(fake_input_dir, tmpdir,
                                              hist_length=64*12*12):
    # Configuration
    test_input_dir = os.path.join(str(tmpdir), 'test_input')
    shutil.copytree(fake_input_dir, test_input_dir)

    # Transformation
    train_files = glob_train_files(test_input_dir)
    adapt_files = glob_adapt_files(test_input_dir, 10000)

    for data_file in adapt_files + train_files:
        X, t = load_h5_input_file(data_file)
        # Corruption
        t = corrupt_t(t)
        os.unlink(data_file)
        save_h5_input_file(data_file, X, t, output_dim = 2)

    # Evaluation
    for eval_step in range(-1, 60, 5):
        original_X = load_data_matrix(fake_input_dir, eval_step=eval_step,
                                                hist_length=hist_length)
        X_with_corrupted_t = load_data_matrix(test_input_dir, eval_step=eval_step,
                                                hist_length=hist_length)
        np.testing.assert_array_equal(original_X, X_with_corrupted_t)


def do_nothing(X,t):
    return X, t

def empty_t(X,t):
    return X, np.array([])

def all_nan_t(X,t):
    return X, np.array([np.nan for _ in t])

def random_corrupt_t(X,t):
    corruption = np.random.choice(4)
    corrupted_t = t.copy()
    if corruption == 0:
        return X, np.array([])
    elif corruption == 1:
        return X, np.array([np.nan for _ in t])
    elif corruption == 2:
        corrupted_t = t.astype('float')
        corrupted_t[np.random.choice(len(t), len(t) // 2, replace=False)] = np.nan
        return X, corrupted_t
    elif corruption == 3:
        corrupted_t[np.random.choice(len(t), len(t) // 2, replace=False)] = 0
        return X, corrupted_t

def shuffle(X,t):
    shuffle_indice = np.random.permutation(len(t))
    corrupted_t = t[shuffle_indice]
    corrupted_X = X[shuffle_indice]
    return corrupted_X, corrupted_t


@pytest.mark.parametrize('corrupt_fn_train, corrupt_fn_adapt', [
    (do_nothing, do_nothing),
    (empty_t, empty_t),
    (all_nan_t, all_nan_t),
    (empty_t, do_nothing),
    (do_nothing, empty_t),
    (random_corrupt_t, random_corrupt_t),
    (shuffle, shuffle),
])
def test_sequential_load_corrupted_t(fake_input_dir, tmpdir, corrupt_fn_train, corrupt_fn_adapt,
                         hist_length=64*12*12):
    # Configuration
    test_input_dir = os.path.join(str(tmpdir), 'test_input')
    shutil.copytree(fake_input_dir, test_input_dir)

    # Transformation
    train_files = glob_train_files(test_input_dir)
    adapt_files = glob_adapt_files(test_input_dir, 10000)

    for data_file in adapt_files + train_files:
        X, t = load_h5_input_file(data_file)
        # Corruption
        if data_file in train_files:
            X, t = corrupt_fn_train(X, t)
        else:
            X, t = corrupt_fn_adapt(X, t)
        os.unlink(data_file)
        save_h5_input_file(data_file, X, t, output_dim = 2)

    # Evaluation
    cache_X = None
    cache_t = None
    cache_X_corrupted = None
    cache_t_corrupted = None
    for eval_step in range(-1, 30, 1):
        original_X, last_time_step, cache_X, cache_t = load_prediction_matrix(
            fake_input_dir, fake_input_dir,
            fake_input_dir, fake_input_dir,
            eval_step, eval_step-1,
            hist_length, cache_X, cache_t)
        corrupted_X, last_time_step, cache_X_corrupted, cache_t_corrupted = load_prediction_matrix(
            test_input_dir, test_input_dir,
            test_input_dir, test_input_dir,
            eval_step, eval_step-1,
            hist_length, cache_X_corrupted, cache_t_corrupted)
        np.testing.assert_array_equal(original_X, corrupted_X)


def test_no_in_h5(fake_input_dir, tmpdir,
                         hist_length=64*12*12):
    # Configuration
    test_input_dir = os.path.join(str(tmpdir), 'test_input')
    shutil.copytree(fake_input_dir, test_input_dir)

    # Transformation
    train_files = glob_train_files(test_input_dir)
    adapt_files = glob_adapt_files(test_input_dir, 10000)

    for data_file in adapt_files + train_files:
        X, t = load_h5_input_file(data_file)

        os.unlink(data_file)
        with h5py.File(data_file, driver = 'core', mode='w') as h5file:
            group = h5file
            group = group.create_group('X')
            group = group.create_group('value')
            group_X = group.create_group('X')
            group_X.create_dataset('value', data = X)

    # Evaluation
    cache_X = None
    cache_t = None
    cache_X_corrupted = None
    cache_t_corrupted = None
    for eval_step in range(-1, 30, 1):
        original_X, last_time_step, cache_X, cache_t = load_prediction_matrix(
            fake_input_dir, fake_input_dir,
            fake_input_dir, fake_input_dir,
            eval_step, eval_step-1,
            hist_length, cache_X, cache_t)
        corrupted_X, last_time_step_2, cache_X_corrupted, cache_t_corrupted = load_prediction_matrix(
            test_input_dir, test_input_dir,
            test_input_dir, test_input_dir,
            eval_step, eval_step-1,
            hist_length, cache_X_corrupted, cache_t_corrupted)
        np.testing.assert_array_equal(original_X, corrupted_X)

        
def test_X_change_shape_no_predict(fake_input_dir, tmpdir,
                 hist_length=64*12*12):
    # Transformation
    train_files = glob_train_files(fake_input_dir)
    adapt_files = glob_adapt_files(fake_input_dir, 10000)

    for data_file in adapt_files + train_files:
        X, t = load_h5_input_file(data_file)
        # Corruption
        if np.random.rand() <= 0.5:
            keep_size = X.shape[1]-300
            X = X[:, :keep_size]
        os.unlink(data_file)
        save_h5_input_file(data_file, X, t, output_dim = 2)
       
    # Evaluation
    cache_X = None
    cache_t = None
    for eval_step in range(-1, 30, 1):
        original_X, last_time_step, cache_X, cache_t = load_prediction_matrix(
            fake_input_dir, fake_input_dir,
            fake_input_dir, fake_input_dir,
            eval_step, eval_step-1,
            hist_length, cache_X, cache_t)
        

def test_compute_ancient_and_recent_timestep_with_nan():
    out = first_non_nan_input(np.array([1300, 1600, 1900, 2200, 2500, 2800]))
    assert out == 0

    out = first_non_nan_input(np.array([np.nan, 1600, 1900, 2200, 2500, 2800]))
    assert out == 1

    out = first_non_nan_input(np.array([np.nan, np.nan, 1900, 2200, 2500, 2800]))
    assert out == 2

    out = first_non_nan_input(np.array([np.nan, np.nan, 1900, 2200, np.nan, np.nan]))
    assert out == 2

    out = first_non_nan_input(np.array([np.nan, 1600, 1900, 2200, 2500, np.nan]))
    assert out == 1

    out = first_non_nan_input(np.array([1300, 1600, 1900, 2200, np.nan, np.nan]))
    assert out == 0

    out = first_non_nan_input(np.array([1300, 1600, 1900, 2200, 2500, np.nan]))
    assert out == 0

    out = first_non_nan_input(np.array([1300, 1600, 1900, np.nan, 2500, np.nan]))
    assert out == 0

    out = first_non_nan_input(np.array([np.nan, 1600, np.nan, 2200, 2500, 2800]))
    assert out == 1

    out = first_non_nan_input(np.array([1300, 1600, 1900, 2200, 2500, 2800]))
    assert out == 0

    out = first_non_nan_input(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]))
    assert np.isnan(out)

    out = most_ancient_timestep_with_nan(np.array([1300, 1600, 1900, 2200, 2500, 2800]))
    assert out == 1300

    out = most_ancient_timestep_with_nan(np.array([np.nan, 1600, 1900, 2200, 2500, 2800]))
    assert out == 1300

    out = most_ancient_timestep_with_nan(np.array([np.nan, np.nan, 1900, 2200, 2500, 2800]))
    assert out == 1300

    out = most_ancient_timestep_with_nan(np.array([np.nan, np.nan, 1900, 2200, np.nan, np.nan]))
    assert out == 1300

    out = most_ancient_timestep_with_nan(np.array([np.nan, 1600, 1900, 2200, 2500, np.nan]))
    assert out == 1300

    out = most_ancient_timestep_with_nan(np.array([1300, 1600, 1900, 2200, np.nan, np.nan]))
    assert out == 1300

    out = most_ancient_timestep_with_nan(np.array([1300, 1600, 1900, 2200, 2500, np.nan]))
    assert out == 1300

    out = most_ancient_timestep_with_nan(np.array([1300, 1600, 1900, np.nan, 2500, np.nan]))
    assert out == 1300

    out = most_ancient_timestep_with_nan(np.array([np.nan, 1600, np.nan, 2200, 2500, 2800]))
    assert out == 1300

    out = most_ancient_timestep_with_nan(np.array([1300, 1600, 1900, 2200, 2500, 2800]))
    assert out == 1300

    out = most_ancient_timestep_with_nan(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]))
    assert np.isnan(out)

    out = most_recent_timestep_with_nan(np.array([1300, 1600, 1900, 2200, 2500, 2800]))
    assert out == 2800

    out = most_recent_timestep_with_nan(np.array([np.nan, 1600, 1900, 2200, 2500, 2800]))
    assert out == 2800

    out = most_recent_timestep_with_nan(np.array([np.nan, np.nan, 1900, 2200, 2500, 2800]))
    assert out == 2800

    out = most_recent_timestep_with_nan(np.array([np.nan, np.nan, 1900, 2200, np.nan, np.nan]))
    assert out == 2800

    out = most_recent_timestep_with_nan(np.array([np.nan, 1600, 1900, 2200, 2500, np.nan]))
    assert out == 2800

    out = most_recent_timestep_with_nan(np.array([1300, 1600, 1900, 2200, np.nan, np.nan]))
    assert out == 2800

    out = most_recent_timestep_with_nan(np.array([1300, 1600, 1900, 2200, 2500, np.nan]))
    assert out == 2800

    out = most_recent_timestep_with_nan(np.array([1300, 1600, 1900, np.nan, 2500, np.nan]))
    assert out == 2800

    out = most_recent_timestep_with_nan(np.array([np.nan, 1600, np.nan, 2200, 2500, 2800]))
    assert out == 2800

    out = most_recent_timestep_with_nan(np.array([1300, 1600, 1900, 2200, 2500, 2800]))
    assert out == 2800

    out = most_recent_timestep_with_nan(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]))
    assert np.isnan(out)
