import time
import os
import os.path
import sys
import argparse
import requests
import logging
import traceback
import subprocess
from threading import Thread

from flask import Flask
from flask import request, jsonify
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import h5py

from src.target_data_io import load_full_cache_matrixes, load_one_more_step, build_predict_matrix_from_cache_and_candidates_timestep, load_h5_input_file
from src.utils import get_taskParameters, get_configs, log_tb_and_force_quit, compact_tb

GLOBAL_STATE = {
    'current_task_id': 0,
    'status': 'idle',
    'eval_step' : None,
    'previous_eval_step' : None,
    'input_dir': None,
    'previous_input_dir': None,
    'output_dir' : None,
    'previous_output_dir' : None,
    'model': None,
    'graph' : None,
    'cache_X' : None,
    'cache_t' : None
}

logger = logging.getLogger('prediction-server')
logger.setLevel(logging.DEBUG)

error_logger = logging.getLogger('error-prediction-server')
error_logger.setLevel(logging.DEBUG)

app = Flask("prediction-server")

def load_model(model_dir, n_gpus=1):
    """Load model from folder"""
    global GLOBAL_STATE

    if GLOBAL_STATE['model'] is None:
        logger.info("Loading model. Using {} gpu".format(n_gpus))
        import tensorflow as tf
        from src.model import EuropaModel
        from src.multi_gpu_utils import multi_gpu_model
        try:
            GLOBAL_STATE['graph'] = tf.get_default_graph()
            with GLOBAL_STATE['graph'].as_default():
                model = EuropaModel.restore_model(model_dir)
                if n_gpus >= 1:
                    model.keras_model = multi_gpu_model(model.keras_model,
                                                        gpus=n_gpus)
                GLOBAL_STATE['model'] = model
        except:
            raise ValueError('Cannot load model from {}'.format(model_dir))
            sys.exit(0)

def load_prediction_matrix(previous_input_dir, input_dir, previous_output_dir, output_dir, eval_step, previous_eval_step, hist_length, cache_X, cache_t):
    condition_load_full_cache = eval_step == 0
    condition_load_full_cache = condition_load_full_cache or (cache_X is None)
    condition_load_full_cache = condition_load_full_cache or (previous_eval_step + 1 != eval_step)
    condition_load_full_cache = condition_load_full_cache or (output_dir != previous_output_dir)
    condition_load_full_cache = condition_load_full_cache or (input_dir != previous_input_dir)

    if condition_load_full_cache:
        logger.info("condition_load_full_cache is True. Loading cache from scratch.  cache_X is None : {}, eval_step : {}, previous_eval_step : {}, input_dir : {}, previous_input_dir : {}, output_dir : {}, previous_output_dir : {}".format(cache_X is None, eval_step, previous_eval_step, input_dir, previous_input_dir, output_dir, previous_output_dir))
        cache_X, cache_t = load_full_cache_matrixes(input_dir, eval_step, hist_length + 100)
    else:
        cache_X, cache_t = load_one_more_step(cache_X, cache_t, input_dir, eval_step, hist_length + 100)

    predict_X = build_predict_matrix_from_cache_and_candidates_timestep(cache_X, cache_t, hist_length)

    return predict_X, cache_t[-1], cache_X, cache_t

def predict_step(input_dir, previous_input_dir, output_dir, previous_output_dir, horizon, eval_step, previous_eval_step, model, graph, cache_X, cache_t):
    'Get prediction for one step and save output'
    try:
        predict_X, last_timestep, cache_X, cache_t = load_prediction_matrix(previous_input_dir, input_dir, previous_output_dir, output_dir, eval_step, previous_eval_step, model.hist_length, cache_X, cache_t)
    except:
        compact_tb(error_logger)
        try:
            X, t = load_h5_input_file(os.path.join(input_dir, 'adapt', 'X0.h5'))
            assert len(X.shape) == 2
            predict_X = np.zeros(shape=(64 * 12 * 12, X.shape[1],))
        except:
            predict_X = np.zeros(shape = (64 * 12 * 12, 1916,))
        last_timestep = 300
        cache_X = None
        cache_t = None

    # Predict and save

    try:
        with graph.as_default():
            pred = model.predict(predict_X)
        if horizon < model.horizon:
            pred = pred[:horizon, :]
        else:
            # predict further horizon with the latest predicted
            pred = np.pad(pred, pad_width=[(0, horizon - model.horizon), (0, 0)],
                          mode='edge')
    except:
        compact_tb(error_logger)
        pred = np.zeros(shape = (horizon, predict_X.shape[1],))

    t_pred = np.arange(horizon)

    return pred, t_pred, cache_X, cache_t

def write_output(output_dir, X, t, eval_step):
    'Write prediction to file'
    path = os.path.join(output_dir, 'Y{}.h5'.format(eval_step))
    with h5py.File(path, 'w', driver = 'core') as f:
        f.create_dataset(name='X', shape=X.shape, data=X)
        f.create_dataset(name='t', shape=t.shape, data=t)

def predict_and_save_to_global_state(eval_step, exit_on_error = True):
    try:
        global GLOBAL_STATE

        # load model
        load_model(GLOBAL_STATE['model_dir'], n_gpus=GLOBAL_STATE['gpu_count'])

        # Refresh horizon
        horizon = get_taskParameters(GLOBAL_STATE['input_dir'])['HORIZON']
        logger.info("Predict step. eval_step : {eval_step}, previous_eval_step : {previous_eval_step}, n_gpus : {n_gpus}, horizon : {horizon}, n_lines_cache : {n_lines_cache}, current_task_id : {current_task_id}".format(
            eval_step = eval_step,
            previous_eval_step = GLOBAL_STATE['previous_eval_step'],
            n_gpus = GLOBAL_STATE['gpu_count'],
            horizon = horizon,
            n_lines_cache = GLOBAL_STATE['cache_X'].shape[1] if GLOBAL_STATE['cache_X'] is not None else "None",
            current_task_id = GLOBAL_STATE['current_task_id']
        ))

        pred, pred_t, cache_X, cache_t = predict_step(
            GLOBAL_STATE['input_dir'],
            GLOBAL_STATE['previous_input_dir'],
            GLOBAL_STATE['output_dir'],
            GLOBAL_STATE['previous_output_dir'],
            horizon,
            eval_step,
            GLOBAL_STATE['previous_eval_step'],
            GLOBAL_STATE['model'],
            GLOBAL_STATE['graph'],
            GLOBAL_STATE['cache_X'],
            GLOBAL_STATE['cache_t'])

        try:
            write_output(GLOBAL_STATE['output_dir'], pred, pred_t, eval_step)
        except:
            compact_tb(error_logger)

        with open(os.path.join(GLOBAL_STATE['code_dir'], 'flask_error_log.txt'), 'r') as f:
            if len(f.read()) > 0:
                os._exit(1)

        GLOBAL_STATE['status'] = 'prediction ended'

        GLOBAL_STATE['previous_eval_step'] = eval_step
        GLOBAL_STATE['previous_input_dir'] = GLOBAL_STATE['input_dir']
        GLOBAL_STATE['previous_output_dir'] = GLOBAL_STATE['output_dir']
        GLOBAL_STATE['cache_X'] = cache_X
        GLOBAL_STATE['cache_t'] = cache_t
        GLOBAL_STATE['current_task_id'] += 1

    except:
        if exit_on_error:
            log_tb_and_force_quit(logger)
        else:
            compact_tb(error_logger)

@app.route('/alive')
def alive():
    return 'alive'

@app.route('/prediction', methods=['POST'])
def prediction():
    "Predict given serialized data"
    try:
        if GLOBAL_STATE['status'] == 'idle':
            data = request.get_json()
            eval_step = data['eval_step']
            GLOBAL_STATE['input_dir'] = data['input_dir']
            GLOBAL_STATE['output_dir'] = data['output_dir']

            thread = Thread(
                target=predict_and_save_to_global_state,
                args=(eval_step,),
            )
            thread.start()

            GLOBAL_STATE['thread'] = thread
            GLOBAL_STATE['status'] = 'prediction running'
            return jsonify({'status': GLOBAL_STATE['status']})
        else:
            return jsonify({'status': 'Error - prediction already running'})
    except:
        log_tb_and_force_quit(logger)

@app.route('/collect')
def collect():
    "Collect end state from latest prediction"
    try:
        if GLOBAL_STATE['status'] == 'prediction ended':
            GLOBAL_STATE['status'] = 'idle'
            return 'finished'
        else:
            return 'not started or finished'
    except:
        log_tb_and_force_quit(logger)

@app.route('/shutdown')
def shutdown():
    "Shutdown server"
    try:
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            return jsonify({'status': 'error - Not running with the Werkzeug Server'})
            # raise RuntimeError('Not running with the Werkzeug Server')
        else:
            func()
            return jsonify({'status': 'shutting down'})
    except:
        log_tb_and_force_quit(logger)


if __name__ == '__main__':
    root_dir = '/home/c4c-user/data'
    parser = argparse.ArgumentParser(description='Start prediction server')
    parser.add_argument('--port', type=int,
                        default=5000, help='port for the server')
    parser.add_argument("--code_dir", help="Code path",
                        type=str, default=os.path.join(root_dir, 'code'))
    parser.add_argument('--gpu_count', type=int, default=1,
                        help='the count of gpu')
    parser.add_argument('--cuda_devices', type=str, default='0',
                        help='the id of the gpu')
    args = parser.parse_args()

    if args.gpu_count >= 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(args.gpu_count)])
        # args.cuda_devices
        # str(args.gpu_id)  # GPU MODE
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""  # CPU MODE

    port = args.port
    try:
        alive_status = requests.get('http://localhost:{}/alive'.format(port)).content
    except:
        alive_status = None


    GLOBAL_STATE['model_dir'] = os.path.join(args.code_dir, 'model')
    GLOBAL_STATE['code_dir'] = args.code_dir
    GLOBAL_STATE['cache_dir'] = os.path.join(args.code_dir, 'cache')
    GLOBAL_STATE['gpu_count'] = args.gpu_count

    if alive_status == b'alive':
        logger.info('flask already running at port {}'.format(port))
        sys.exit(0)
    else:
        subprocess.run(['rm', '-f', os.path.join(GLOBAL_STATE['code_dir'], 'flask_error_log.txt')])
        subprocess.run(['rm', '-f', os.path.join(GLOBAL_STATE['code_dir'], 'flask_log.txt')])
        logger.addHandler(logging.FileHandler(os.path.join(GLOBAL_STATE['code_dir'], 'flask_log.txt')))
        error_logger.addHandler(logging.FileHandler(os.path.join(GLOBAL_STATE['code_dir'], 'flask_error_log.txt')))

        app.run(port=port, debug=False)
