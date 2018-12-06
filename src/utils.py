import os
import sys
import warnings
warnings.filterwarnings('ignore')
from .target_data_io import load_h5_input_file
import numpy as np
import h5py
import traceback
import re
import textwrap

def log_tb_and_force_quit(logger, limit = 10):
    logger.error("\n".join(traceback.format_exc().split('\n')[-limit:]))
    os._exit(1)

def compact_tb(logger):
    full_trace = traceback.format_exc().split('\n')
    compact_tb = compact_trace(full_trace, max_row_size = 110, max_line = 3)
    logger.error(compact_tb)

def compact_trace(full_trace, max_row_size = 110, max_line = 3):
    try:
        exception = full_trace[-2]

        trace = full_trace[1 : -2]
        trace = [trace[i] for i in range(len(trace)) if i % 2 == 0]

        compact_trace = ""

        for call in reversed(trace):
            filename, line = re.findall("File \"(.*)\", line ([\\d]+)", call)[0]

            compact_filename = "/".join(filename.split('/')[-3:]) if '/' in filename else filename

            compact_trace += compact_filename + ' : ' + line + ', '

        compact_trace += " " + exception.strip()

        return "\n".join(textwrap.wrap(compact_trace, max_row_size)[-max_line:])

    except:
        full_trace = [t.strip() for t in full_trace]
        return "\n".join(textwrap.wrap(" ".join(full_trace), max_row_size)[-max_line:])


def get_taskParameters(input_dir):
    '''Parse taskParameters.ini from data folder'''
    taskParameters = {}
    with open(input_dir + '/taskParameters.ini', 'r') as f:
        for l in f.readlines():
            conf, value = l.split('=')
            taskParameters[conf] = int(value)

    for key in ['HORIZON', 'NUMBEROFSTEPS']:
        try:
            assert key in taskParameters.keys()
        except:
            raise ValueError('taskParameters.ini does not contains {}'.format(key))
    return taskParameters


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


def compute_score(solution_dir, result_dir, **parameters):
    score = np.zeros(parameters['NUMBEROFSTEPS'])
    ext = '.h5'
    for step_num in range(parameters['NUMBEROFSTEPS']):
        sfile = 'X' + str(step_num + 1)
        pfile = 'Y' + str(step_num)
        # Get the solution from the res subdirectory (must end with ext)
        solution_file = os.path.join(solution_dir, sfile + ext)
        predict_file = os.path.join(result_dir, pfile + ext)

        # Read the solution and prediction values into numpy arrays
        y_true, _ = load_h5_input_file(solution_file)
        with h5py.File(predict_file) as f:
            y_pred = np.array(f['X'][:])

        horizon = y_pred.shape[0]

        if y_true.shape[0] < horizon:
            try:
                next_obs, _ = load_h5_input_file(os.path.join(solution_dir,
                                                              'X{}.h5'.format(step_num+1)))
                y_true = np.concatenate([y_true, next_obs])
            except:
                y_true = np.pad(y_true, [(0, horizon - y_true.shape[0]), (0,0)],
                                mode='edge')
        y_true = y_true[:horizon]
        score[step_num] = mse(y_true, y_pred)
    return score


def get_configs(args, config_eval_step = True):
    """Combine different sources of configs"""
    configs = {}
    taskParameters = get_taskParameters(args.input_dir)

    configs.update(
        dict(
            input_dir=os.path.abspath(args.input_dir),
            output_dir=os.path.abspath(args.output_dir),
            code_dir=os.path.abspath(args.code_dir),
            **taskParameters
        )
    )

    if config_eval_step:
        configs.update(
            dict(
                eval_step=args.eval_step
            )
        )

    configs['cache_dir'] = os.path.join(configs['code_dir'], 'cache')
    configs['model_dir'] = os.path.join(configs['code_dir'], 'model')
    return configs
