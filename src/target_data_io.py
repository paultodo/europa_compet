import pandas as pd
import numpy as np
import glob
import warnings
warnings.filterwarnings('ignore')
import h5py
import re
import os
import logging

logger = logging.getLogger('prediction-server')

def glob_train_files(input_dir):
    train_path = os.path.join(input_dir, 'train')

    train_folders = glob.glob(os.path.join(train_path, 'Xm[1-4]'))
    train_folders = [*filter(lambda x: re.search(r'.*Xm([\d]+)$', x), train_folders)]
    train_folders = sorted(train_folders, key = lambda x : int(re.findall(r'.*Xm([\d]+)$', x)[0]))

    all_train_files = []
    for train_folder in train_folders:
        train_files = glob.glob(os.path.join(train_folder, 'X*.h5'))
        train_files = [*filter(lambda x: re.search(r'.*X([\d]+).h5$', x), train_files)]
        train_files = sorted(train_files, key = lambda x : int(re.findall(r'.*X([\d]+).h5$', x)[0]))

        all_train_files += train_files

    return all_train_files

def glob_adapt_files(input_dir, eval_step):
    adapt_path = os.path.join(input_dir, 'adapt')

    all_adapt_files = glob.glob(os.path.join(adapt_path, 'X*.h5'))
    all_adapt_files = [*filter(lambda x: re.search(r'.*X([\d]+).h5$', x), all_adapt_files)]
    all_adapt_files = sorted(all_adapt_files, key = lambda x : int(re.findall(r'.*X([\d]+).h5$', x)[0]))
    all_adapt_files = [file for file in all_adapt_files if int(re.findall(r'.*X([\d]+).h5$', file)[0]) <= eval_step]

    return all_adapt_files

def fillna_t(t):
    t = t.copy()
    na_count = np.sum(np.isnan(t))
    if na_count == 0:
        return t
    elif na_count == t.shape[0]:
        return np.arange(t.shape[0]) * 300
    else:
        for i in range(1, len(t)):  # front fill
            t[i] = t[i-1] + 300 if np.isnan(t[i]) else t[i]
        for i in range(len(t)-1, -1, -1):  # then back fill
            t[i] = t[i+1] - 300 if np.isnan(t[i]) else t[i]
    assert np.sum(np.isnan(t)) == 0
    return t


def load_h5_input_file(file):
    try:
        # Loading file
        h5file = h5py.File(file, mode='r', driver = 'core', backing_store = False)
    except:
        logger.info("File {}. The loading of the h5 file failed. Not using dataset".format(file))
        return None, None

    try:
        # Processing X matrix
        X = np.array(h5file['X']['value']['X']['value'])

        if len(X.shape) >= 3:
            if len(X.shape) >= 4:
                logger.info("File {} has {} dimensions for X. Shape is {}. Reshaping".format(file, len(X.shape), X.shape))
            for _ in range(len(X.shape) - 2):
                X = X[..., 0]

        X = np.nan_to_num(X)

        if X.shape[0] == 0:
            logger.info("File {} has no point for X. Shape is {}. Removing".format(file, X.shape))

            return None, None
    except:
        logger.info("File {}. The loading of X failed. Not using dataset".format(file))
        return None, None

    # Processing t matrix
    try:
        t = np.array(h5file['X']['value']['t']['value'])
        t = t.reshape(-1)
        t = t.astype('float')

        # Shape of t
        if t.shape[0] > X.shape[0]:
            logger.info("File {}. t was too big. t_shape {}, X_shape {}. Pruning".format(file, t.shape, X.shape))
            t = t[:X.shape[0]]

        if t.shape[0] < X.shape[0]:
            logger.info("File {}. t was too small. t_shape {}, X_shape {}. Adding".format(file, t.shape, X.shape))
            t = np.append(t, np.repeat(np.nan, X.shape[0] - t.shape[0]), axis = 0)

        # fill 0 to nan
        if np.any(t == 0):
            t[t == 0] = np.nan
            logger.info("File {}. t had some 0 values. Filling to nan".format(file))

    except:
        logger.info("File {}. The loading of t failed. Using nan for timesteps".format(file))
        t = np.empty(X.shape[0])
        t[:] = np.nan

    return X, t

def save_h5_input_file(file_path, X, t, output_dim = 3):
    try:
        np.testing.assert_array_equal(np.sort(t.ravel()), t.ravel(), "The timestep array needs to be sorted")
    except:
        Warning('Timesteps are not sorted. Saving anyway')
    if len(X.shape) >= 2:
        if output_dim == 3:
            X_shape = (X.shape[0], X.shape[1], 1,)
        else:
            X_shape = (X.shape[0], X.shape[1],)
        X = X.reshape(*X_shape)

    h5file = h5py.File(file_path, driver = 'core')
    group = h5file
    group = group.create_group('X')
    group = group.create_group('value')
    group_X = group.create_group('X')
    group_X.create_dataset('value', data = X)

    group_t = group.create_group('t')
    group_t.create_dataset('value', data = t.astype(float).reshape(-1, 1))

def first_non_nan_input(array):
    assert len(array.shape) == 1
    is_nan_array = np.isnan(array)
    unique_values, unique_first_index = np.unique(is_nan_array, return_index = True)
    if unique_first_index.shape[0] == 1:
        if unique_values[0] == 1:
            return np.nan
        else:
            return 0
    else:
        first_not_nan_array = unique_first_index[0]
        return first_not_nan_array

def most_ancient_timestep_with_nan(array):
    assert len(array.shape) == 1
    non_nan_index = first_non_nan_input(array)

    if np.isnan(non_nan_index):
        return np.nan
    else:
        return array[non_nan_index] - 300 * (non_nan_index)

def most_recent_timestep_with_nan(array):
    assert len(array.shape) == 1
    non_nan_index = array.shape[0] - 1 - first_non_nan_input(array[::-1])

    if np.isnan(non_nan_index):
        return np.nan
    else:
        return array[non_nan_index] + 300 * (array.shape[0] - 1 - non_nan_index)

def load_full_cache_matrixes(input_dir, eval_step, steps_needed):
    all_train_files = glob_train_files(input_dir)
    all_adapt_files = glob_adapt_files(input_dir, eval_step)

    all_files = all_train_files + all_adapt_files

    all_X = None
    all_t = None

    for file in reversed(all_files):  # read the files from the most recent to the most ancient
        X, t = load_h5_input_file(file)

        # If the loading failed
        if X is None:
            continue
        
        try:
            all_X = np.append(X, all_X, axis = 0) if all_X is not None else X
            all_t = np.append(t, all_t, axis = 0) if all_t is not None else t
        except:
            diff_dim = all_X.shape[1] - X.shape[1]
            if diff_dim > 0:
                X = np.pad(X, pad_width=((0, 0), (0, diff_dim)),
                              mode='constant', constant_values=0)
            elif diff_dim < 0:
                X = X[:, :all_X.shape[1]]
            try:
                all_X = np.append(X, all_X, axis = 0) if all_X is not None else X
                all_t = np.append(t, all_t, axis = 0) if all_t is not None else t
            except:
                raise ValueError('Shape not good X {}, all_X {}'.format(str(X.shape), str(all_X.shape)))

        
        if all_t is not None:
            most_ancient_timestep = most_ancient_timestep_with_nan(all_t)
            most_recent_timestep = most_recent_timestep_with_nan(all_t)

            if np.isnan(most_ancient_timestep) and np.isnan(most_recent_timestep) and all_t.shape[0] > steps_needed:
                # When all is nan but there is enough points
                break

            if most_ancient_timestep + steps_needed * 300 < most_recent_timestep:
                # We have enough data to make a predict
                break

    if all_X is None:
        logger.info("All loads failed ! What can we return !?")

    return all_X, all_t

def get_necessary_steps(t, history_length):
    # This is the timesteps normalized. A distance of 1 is equivalent to 5 minutes.
    # We are trying to extract 'history_length' timesteps from our matrix X.
    # This normed array gives a floating/approximated position in the matrix X.
    # There can be multiple floating position candidate for each integer timestep position in [0, history_length[
    # e.g : 0.95 and 1.1 are two candidates positions for the integer timestep position 1
    t_normed = (t - t[-1]) / 300. + history_length - 1

    # This quanity is the integer timestep position that can be filled by the floating positions.
    t_round = np.round(t_normed).astype(int)

    # This quantity gives the distance of the floating positions from their integer position
    t_delta_abs = np.abs(t_normed - t_round)

    # This quantity is the indices sorted by integer timestep, then by distance between the floating position
    # and the integer position. So that when there is two candidates like 0.95 and 1.1 for the position 1,
    # the candidate with the smallest distance will be chosen, which here is 0.95
    indices_sorted = np.lexsort([t_delta_abs, t_round])

    # indices_unique is the location of the first indice where a new integer timestep was reached in t_round
    timesteps_inside_t, indices_unique = np.unique(t_round, return_index = True)

    # These are the indices which we should keep, because they are the best candidates for each integer timesteps
    indices_kept = indices_sorted[indices_unique]

    # These are the integer timesteps we keps using the indices
    timesteps_integer_kept = t_round[indices_kept]

    # We often have too much history so we prune it
    indices_kept = indices_kept[timesteps_integer_kept >= 0]
    timesteps_integer_kept = timesteps_integer_kept[timesteps_integer_kept >= 0]

    return indices_kept, timesteps_integer_kept

def build_predict_matrix_from_cache_and_candidates_timestep(X, t, history_length):
    filled_t = fillna_t(t)

    # Drop duplicates in t if exists, ensure t is unique and ordered:
    timesteps_unique, indices_unique = np.unique(filled_t, return_index = True)

    sorted_X = X[indices_unique]
    sorted_filled_t = filled_t[indices_unique]

    indices_kept, timesteps_integer_keps = get_necessary_steps(sorted_filled_t, history_length)

    predict_X = np.zeros((history_length, sorted_X.shape[1],))

    predict_X[timesteps_integer_keps] = sorted_X[indices_kept]

    return predict_X

def load_one_more_step(cache_X, cache_t, input_dir, eval_step, steps_needed):
    adapt_folder = os.path.join(input_dir, 'adapt')
    adapt_file = os.path.join(adapt_folder, 'X{}.h5'.format(eval_step))

    X, t = load_h5_input_file(adapt_file)

    # Case file was not loaded properly
    if X is None:
        return cache_X, cache_t

    # Case count of HV lines are different
    if X.shape[1] != cache_X.shape[1]:
        logger.info("There is not the same amount of HV lines between cache and adapt. Reload from scratch. n_lines_cache : {}, n_lines_new : {}, eval_step {}".format(cache_X.shape[1], X.shape[1], eval_step))
        cache_X, cache_t = load_full_cache_matrixes(input_dir, eval_step, steps_needed)

        return cache_X, cache_t

    cache_X = np.append(cache_X, X, axis = 0)
    cache_t = np.append(cache_t, t, axis = 0)

    cache_t_filled = fillna_t(cache_t)

    most_recent_timestep = cache_t_filled[-1]

    cache_X = cache_X[cache_t_filled + steps_needed * 300 >= most_recent_timestep]
    cache_t = cache_t[cache_t_filled + steps_needed * 300 >= most_recent_timestep]

    return cache_X, cache_t
