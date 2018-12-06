import pytest
import os
import shutil
import subprocess
import requests
#os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Avoid gpu oom
from start_prediction_server import load_prediction_matrix
from src.target_data_io import glob_adapt_files, glob_train_files, save_h5_input_file, load_h5_input_file
import numpy as np
from config import USE_FLASK


def send_shutdown(server_full_address):
    try:
        answer = requests.get('{}/shutdown'.format(server_full_address)).json()
        assert answer['status'] == 'shutting down'
    except AssertionError:
        raise AssertionError('Expecting status "shutting down", instead got {}'.format(answer))
    except Exception as e:
        raise e


def run_predict_on_test_dataset_and_output_score(name, code_dir, fake_input_dir, test_input_dir, test_output_dir, tests_log_path, run_compute_score=True):
    cmd = [
        'bash',
        os.path.join(code_dir, 'run_all.sh'),
        test_input_dir,
        test_output_dir,
        code_dir,
    ]

    os.makedirs(test_output_dir, exist_ok = True)
    with open(os.path.join(test_output_dir, 'trash_log.txt'), 'w') as fout:
        subprocess.run(cmd)#, stdout = fout, stderr = fout)
    
    if run_compute_score:
        cmd = [
            'python3',
            os.path.join(code_dir, 'compute_score.py'),
            '--input_dir', fake_input_dir,
            '--output_dir', test_output_dir,
        ]

        subprocess.run(cmd)

        with open(os.path.join(test_output_dir, 'score.txt'), 'r') as f:
            scores = f.read()

        with open(tests_log_path, 'a') as fout:
            fout.write(name + '\n')
            fout.write(scores)
            fout.write('\n\n')

    if USE_FLASK:
        send_shutdown('http://localhost:4130')



def remove_timesteps(X, t, cumulated_steps):
    np.random.seed(0)
    index = np.append(np.random.choice(np.arange(X.shape[0], dtype = int), replace = False, size = 9 * X.shape[0] // 10), [X.shape[0] - 1])
    index = np.sort(np.unique(index))
    X = X[index]
    t = t[index]
    assert X.shape[0] == t.shape[0]
    return X, t

def test_missing_timesteps(tmpdir, fake_input_dir, output_dir, code_dir):
    # Configuration
    test_input_dir = os.path.join(str(tmpdir), 'test_input')
    test_output_dir = os.path.join(str(tmpdir), 'test_output')
    tests_log_path = os.path.join(code_dir, 'tests_scores_log.txt')
    shutil.copytree(fake_input_dir, test_input_dir)

    # Transformation
    adapt_files = glob_adapt_files(test_input_dir, 10000)

    for adapt_file in adapt_files:
        X, t = load_h5_input_file(adapt_file)
        # Corruption
        X, t = remove_timesteps(X, t, None)
        os.unlink(adapt_file)
        save_h5_input_file(adapt_file, X, t, output_dim = 2)

    # Evaluation
    run_predict_on_test_dataset_and_output_score('test_missing_timesteps', code_dir, fake_input_dir, test_input_dir, test_output_dir, tests_log_path)





def fill_with_0(X, t, cumulated_steps):
    np.random.seed(0)
    index = np.append(np.random.choice(np.arange(X.shape[0], dtype = int), replace = False, size = 9 * X.shape[0] // 10), [X.shape[0] - 1])
    index = np.sort(np.unique(index))

    new_X = np.zeros(shape = X.shape)
    new_X[index] = X[index]
    assert X.shape[0] == new_X.shape[0]
    return new_X, t

def test_fill_timesteps_with_zeros(tmpdir, fake_input_dir, output_dir, code_dir):
    # Configuration
    test_input_dir = os.path.join(str(tmpdir), 'test_input')
    test_output_dir = os.path.join(str(tmpdir), 'test_output')
    tests_log_path = os.path.join(code_dir, 'tests_scores_log.txt')
    shutil.copytree(fake_input_dir, test_input_dir)

    # Transformation
    adapt_files = glob_adapt_files(test_input_dir, 10000)

    for adapt_file in adapt_files:
        X, t = load_h5_input_file(adapt_file)
        # Corruption
        X, t = fill_with_0(X, t, None)
        os.unlink(adapt_file)
        save_h5_input_file(adapt_file, X, t, output_dim = 2)

    # Evaluation
    run_predict_on_test_dataset_and_output_score('test_fill_timesteps_with_zeros', code_dir, fake_input_dir, test_input_dir, test_output_dir, tests_log_path)






def fill_with_nan(X, t, cumulated_steps):
    np.random.seed(0)
    index = np.append(np.random.choice(np.arange(X.shape[0], dtype = int), replace = False, size = 9 * X.shape[0] // 10), [X.shape[0] - 1])
    index = np.sort(np.unique(index))

    new_X = np.empty(shape = X.shape)
    new_X[:] = np.nan
    new_X[index] = X[index]
    assert X.shape[0] == new_X.shape[0]
    return new_X, t

def test_fill_timesteps_with_nan(tmpdir, fake_input_dir, output_dir, code_dir):
    # Configuration
    test_input_dir = os.path.join(str(tmpdir), 'test_input')
    test_output_dir = os.path.join(str(tmpdir), 'test_output')
    tests_log_path = os.path.join(code_dir, 'tests_scores_log.txt')
    shutil.copytree(fake_input_dir, test_input_dir)

    # Transformation
    adapt_files = glob_adapt_files(test_input_dir, 10000)

    for adapt_file in adapt_files:
        X, t = load_h5_input_file(adapt_file)
        # Corruption
        X, t = fill_with_nan(X, t, None)
        os.unlink(adapt_file)
        save_h5_input_file(adapt_file, X, t, output_dim = 2)

    # Evaluation
    run_predict_on_test_dataset_and_output_score('test_fill_timesteps_with_nan', code_dir, fake_input_dir, test_input_dir, test_output_dir, tests_log_path)



def slightly_move_timesteps(X, t):
    np.random.seed(0)

    move = np.round(np.random.uniform(low = -50, high = 50, size = t.shape[0])).astype(int)

    new_t = t + move

    return X, new_t

def test_fill_slightly_move_timesteps(tmpdir, fake_input_dir, output_dir, code_dir):
    # Configuration
    test_input_dir = os.path.join(str(tmpdir), 'test_input')
    test_output_dir = os.path.join(str(tmpdir), 'test_output')
    tests_log_path = os.path.join(code_dir, 'tests_scores_log.txt')
    shutil.copytree(fake_input_dir, test_input_dir)

    # Transformation
    adapt_files = glob_adapt_files(test_input_dir, 10000)

    for adapt_file in adapt_files:
        X, t = load_h5_input_file(adapt_file)
        # Corruption
        X, t = slightly_move_timesteps(X, t)
        os.unlink(adapt_file)
        save_h5_input_file(adapt_file, X, t, output_dim = 2)

    # Evaluation
    run_predict_on_test_dataset_and_output_score('test_fill_slightly_move_timesteps', code_dir, fake_input_dir, test_input_dir, test_output_dir, tests_log_path)





@pytest.mark.parametrize('horizon', [
    1, 11, 15, 24
])
def test_other_horizon(tmpdir, fake_input_dir, output_dir, code_dir, horizon):
    # Configuration
    test_input_dir = os.path.join(str(tmpdir), 'test_input')
    test_output_dir = os.path.join(str(tmpdir), 'test_output')
    tests_log_path = os.path.join(code_dir, 'tests_scores_log.txt')
    shutil.copytree(fake_input_dir, test_input_dir)

    with open(os.path.join(test_input_dir, 'taskParameters.ini'), 'w') as fout:
        fout.write('NUMBEROFSTEPS=68\nHORIZON={}'.format(horizon))

    # Evaluation
    run_predict_on_test_dataset_and_output_score('test_other_horizon_{}'.format(horizon), code_dir, fake_input_dir, test_input_dir, test_output_dir, tests_log_path)




def remove_half_lines(X, t):
    new_X = X[:, X.shape[1] // 2:]

    return new_X, t

def test_remove_half_lines(tmpdir, fake_input_dir, output_dir, code_dir):
    # Configuration
    test_input_dir = os.path.join(str(tmpdir), 'test_input')
    test_output_dir = os.path.join(str(tmpdir), 'test_output')
    tests_log_path = os.path.join(code_dir, 'tests_scores_log.txt')
    shutil.copytree(fake_input_dir, test_input_dir)

    # Transformation
    train_files = glob_train_files(test_input_dir)
    adapt_files = glob_adapt_files(test_input_dir, 10000)

    for data_file in adapt_files + train_files:
        X, t = load_h5_input_file(data_file)
        # Corruption
        X, t = remove_half_lines(X, t)
        os.unlink(data_file)
        save_h5_input_file(data_file, X, t, output_dim = 2)

    # Evaluation
    run_predict_on_test_dataset_and_output_score('test_remove_half_lines', code_dir, test_input_dir, test_input_dir, test_output_dir, tests_log_path)


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


def test_corrupted_t(tmpdir, fake_input_dir, output_dir, code_dir):
    # Configuration
    test_input_dir = os.path.join(str(tmpdir), 'test_input')
    test_output_dir = os.path.join(str(tmpdir), 'test_output')
    tests_log_path = os.path.join(code_dir, 'tests_scores_log.txt')
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
    run_predict_on_test_dataset_and_output_score('test_corrupted_t', code_dir, fake_input_dir, test_input_dir, test_output_dir, tests_log_path)


def test_empty_X(tmpdir, fake_input_dir, output_dir, code_dir):
    # Configuration
    test_input_dir = os.path.join(str(tmpdir), 'test_input')
    test_output_dir = os.path.join(str(tmpdir), 'test_output')
    tests_log_path = os.path.join(code_dir, 'tests_scores_log.txt')
    shutil.copytree(fake_input_dir, test_input_dir)

    # Transformation
    train_files = glob_train_files(test_input_dir)
    adapt_files = glob_adapt_files(test_input_dir, 10000)

    for data_file in adapt_files + train_files:
        X, t = load_h5_input_file(data_file)
        # Corruption
        if np.random.rand() <= 0.1:
            X = np.array([])
        os.unlink(data_file)
        save_h5_input_file(data_file, X, t, output_dim = 2)

    # Evaluation
    run_predict_on_test_dataset_and_output_score('test_empty_X_10%', code_dir, fake_input_dir, test_input_dir, test_output_dir, tests_log_path)

    
def test_X_change_shape(tmpdir, fake_input_dir, output_dir, code_dir):
    # Configuration
    test_input_dir = os.path.join(str(tmpdir), 'test_input')
    test_output_dir = os.path.join(str(tmpdir), 'test_output')
    tests_log_path = os.path.join(code_dir, 'tests_scores_log.txt')
    shutil.copytree(fake_input_dir, test_input_dir)

    # Transformation
    train_files = glob_train_files(test_input_dir)
    adapt_files = glob_adapt_files(test_input_dir, 10000)

    for data_file in adapt_files + train_files:
        X, t = load_h5_input_file(data_file)
        # Corruption
        if np.random.rand() <= 0.5:
            keep_size = X.shape[1]-300
            X = X[:, :keep_size]
        os.unlink(data_file)
        save_h5_input_file(data_file, X, t, output_dim = 2)

    # Evaluation
    run_predict_on_test_dataset_and_output_score('test_X_change_shape', code_dir, fake_input_dir, test_input_dir, test_output_dir, tests_log_path, run_compute_score=False)