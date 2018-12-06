import numpy as np
import os
import time
import requests
import json
import tempfile

DEBUG = False

def wait_flask_alive(server_full_address, check_interval = 0.05, timeout = 60):
    "Wait until flask server is alive"
    timeout_counter = timeout / check_interval
    while True:
        try:
            response = requests.get('{}/alive'.format(server_full_address))
            answer = response.content
            if answer == b'alive':
                break
        except:
            time.sleep(check_interval)
            timeout_counter -= 1

            if timeout_counter <= 0:
                raise TimeoutError("Timeout in waiting flask alive. Raising exception")

def send_flask_prediction(server_full_address, eval_step, input_dir, output_dir, debug = False):
    "Send prediction request to server"
    response = requests.post('{}/prediction'.format(server_full_address),
                             data=json.dumps({'eval_step': eval_step, "input_dir" : input_dir, "output_dir": output_dir}),
                             headers={"Content-type": "application/json"})
    if debug:
        print(response.content)
    answer = response.json()
    try:
        assert answer['status'] == 'prediction running' # Y a rien qui catch ca
    except:
        raise AssertionError('Expecting status "prediction running", instead got {}'.format(answer['status']))

def wait_flask_collect(server_full_address, check_interval = 0.05, debug = False, timeout=30):
    "Wait for prediction"
    collect_address = '{}/collect'.format(server_full_address)
    timeout_counter = timeout / check_interval
    while True:
        response = requests.get(collect_address)
        answer = response.content
        if answer == b'finished':
            return
        else:
            time.sleep(check_interval)
            timeout_counter -= 1
            if timeout_counter <= 0:
                raise TimeoutError("Timeout in waiting for prediction result. Raising exception")

def predict(server_full_address, eval_step, input_dir, output_dir, timeout=30):
    start_time = time.time()
    wait_flask_alive(server_full_address, timeout=timeout)
    send_flask_prediction(server_full_address, eval_step, input_dir, output_dir)
    remaining_time = int(timeout - (time.time() - start_time))
    wait_flask_collect(server_full_address, timeout=remaining_time)
    try:
        assert os.path.isfile(
            os.path.join(output_dir, 'Y{}.h5'.format(eval_step))
        )
    except:
        raise AssertionError('Server reports finished but prediction file not found')


def full_address(ip_address, port):
    return 'http://{}:{}'.format(ip_address, port)
