import argparse
import os
import sys
from prepare_model import prepare_model
from src.utils import get_configs, log_tb_and_force_quit
from src.flask_util import full_address, predict, wait_flask_alive
import subprocess
import logging

from config import USE_FLASK
import start_prediction_server
from start_prediction_server import predict_and_save_to_global_state

DEBUG = False


def start_prediction_server_function(configs, gpu_count=1, port=4130, return_process=False):
    cmd = 'nohup python3 {code_dir}/start_prediction_server.py \
            --code_dir {code_dir} \
            --port {port} \
            --gpu_count {gpu_count} > /dev/null &'.format(
                code_dir=configs['code_dir'],
                model_dir=configs['model_dir'],
                port=port,
                gpu_count=gpu_count)
    if not return_process:
        subprocess.run(cmd, shell=True)
    else:
        process = subprocess.Popen(cmd, shell=True)
        return process

def predict(input_dir, output_dir, code_dir, gpu_count, eval_step):
    subprocess.run(['rm', '-f', os.path.join(code_dir, 'flask_error_log.txt')])
    start_prediction_server.logger.addHandler(logging.FileHandler(os.path.join(code_dir, 'flask_log.txt')))
    start_prediction_server.error_logger.addHandler(logging.FileHandler(os.path.join(code_dir, 'flask_error_log.txt')))

    if gpu_count >= 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(gpu_count)])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""  # CPU MODE

    start_prediction_server.GLOBAL_STATE['input_dir'] = input_dir
    start_prediction_server.GLOBAL_STATE['output_dir'] = output_dir
    start_prediction_server.GLOBAL_STATE['model_dir'] = os.path.join(code_dir, 'model')
    start_prediction_server.GLOBAL_STATE['code_dir'] = code_dir
    start_prediction_server.GLOBAL_STATE['cache_dir'] = os.path.join(code_dir, 'cache')
    start_prediction_server.GLOBAL_STATE['gpu_count'] = gpu_count

    predict_and_save_to_global_state(eval_step, exit_on_error = False)


if __name__ == '__main__': # Default values based on starting kit v1.01 
    parser = argparse.ArgumentParser()
    root_dir = '/home/c4c-user/data'
    parser.add_argument('--eval_step', help='Evaluation step. E.g. 0,1,2',
                        type=int, default=0)
    parser.add_argument("--input_dir", help="Input data path, must contain \
                        taskParameters.ini, train/ and adapt/",
                        type=str, default=os.path.join(root_dir, 'sample_data'))
    parser.add_argument("--output_dir", help="Output data path, will store prediction",
                        type=str, default=os.path.join(root_dir, 'result'))
    parser.add_argument("--code_dir", help="Code path",
                        type=str, default=os.path.join(root_dir, 'code'))
    parser.add_argument("--use_flask", help="Use flask server for faster prediction",
                        type=int, default=0)
    parser.add_argument("--gpu_count", help="Default count of GPU",
                        type=int, default=1)
    args = parser.parse_args()
    configs = get_configs(args)
    sys.path.append(configs.get('code_dir'))

    # Prepare folders if needed
    os.makedirs(configs['cache_dir'], exist_ok=True)
    os.makedirs(configs['model_dir'], exist_ok=True)
    os.makedirs(configs['output_dir'], exist_ok=True)

    if USE_FLASK:
        try:
            wait_flask_alive(timeout=0.1, check_interval=0.01)
        except:
            # Prepare model for prediction
            prepare_model(configs)
            # Launch server if use flask
            start_prediction_server_function(configs, gpu_count=args.gpu_count, port = 4130)
    
        # Predict
        server_full_address = full_address('127.0.0.1', port = 4130)
        try:
            predict(server_full_address, configs['eval_step'],
                    configs['input_dir'], configs['output_dir'],
                    timeout=600)
        except:
            flask_log_file = os.path.join(configs['code_dir'], 'flask_log.txt')
            flask_error_log_file = os.path.join(configs['code_dir'], 'flask_error_log.txt')
            if os.path.isfile(flask_log_file):
                with open(flask_log_file, 'r') as f:
                    log = "".join(f.readlines()[-4:])
            else:
                log = 'Could not open flask log file. Flask had an exception before the logging handler.'
    
            if os.path.isfile(flask_error_log_file):
                with open(flask_error_log_file, 'r') as f:
                    error_log = f.read()
    
                if len(error_log) > 0:
                    log = log + "\n" + error_log
    
            if configs['eval_step'] > 1:
                os.unlink(os.path.join(configs['output_dir'], "Y{}.h5".format(configs['eval_step'])))
    
            logger = logging.getLogger()
            logger.addHandler(logging.StreamHandler(sys.stderr))
    
            #logger.error('-' * 20 + 'Flask log' + '-' * 20 + '\n')
            logger.error(log)
            #logger.error('-' * 20 + 'Main log' + '-' * 20 + '\n')
            log_tb_and_force_quit(logger, limit = 4)
            #logger.error('-' * 20 + 'End log' + '-' * 20 + '\n')
    
            if configs['eval_step'] > 1:
                sys.exit(1)
    else:
        # Prepare model for prediction
        prepare_model(configs)

        predict(configs['input_dir'], configs['output_dir'], configs['code_dir'], 1, configs['eval_step'])

        flask_log_file = os.path.join(configs['code_dir'], 'flask_log.txt')
        flask_error_log_file = os.path.join(configs['code_dir'], 'flask_error_log.txt')
        if os.path.isfile(flask_log_file):
            with open(flask_log_file, 'r') as f:
                log = "".join(f.readlines()[-4:])
        else:
            log = 'Could not open flask log file. Flask had an exception before the logging handler.'

        if os.path.isfile(flask_error_log_file):
            with open(flask_error_log_file, 'r') as f:
                error_log = f.read()

            if len(error_log) > 0:
                log = log + "\n" + error_log

                os.unlink(os.path.join(configs['output_dir'], "Y{}.h5".format(configs['eval_step'])))

                logger = logging.getLogger()
                logger.addHandler(logging.StreamHandler(sys.stderr))

                logger.error(log)
        
                sys.exit(1)
