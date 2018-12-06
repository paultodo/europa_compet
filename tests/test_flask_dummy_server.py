import subprocess
import pytest
import requests
from src.flask_util import wait_flask_alive, send_flask_prediction, wait_flask_collect, predict


def send_shutdown(server_full_address):
    try:
        answer = requests.get('{}/shutdown'.format(server_full_address)).json()
        assert answer['status'] == 'shutting down'
    except AssertionError:
        raise AssertionError('Expecting status "shutting down", instead got {}'.format(answer))
    except Exception as e:
        raise e


@pytest.fixture
def flask_dummy_server(code_dir, model_dir):
    try:
        send_shutdown('http://localhost:4130')
    except requests.exceptions.ConnectionError:
        pass
    except Exception as e:
        raise e
    process = subprocess.Popen(
        'python3 {}/tests/dummy_flask_server.py'.format(code_dir),
        shell=True)
    yield 'http://localhost:4130'
    try:
        send_shutdown('http://localhost:4130')
        process.terminate()
    except:
        pass


def test_wait_flask_alive_ok(flask_dummy_server):
    wait_flask_alive(flask_dummy_server, timeout=10)


def test_shutdown_request_ok(flask_dummy_server):
    wait_flask_alive(flask_dummy_server)
    send_shutdown(flask_dummy_server)
    with pytest.raises(Exception):
        wait_flask_alive(flask_dummy_server, timeout=10)


def test_shutdown_correctly(flask_dummy_server):
    def dummy_test(flask_dummy_server):
        pass
    dummy_test(flask_dummy_server)
    with pytest.raises(Exception):
        wait_flask_alive('localhost:4130', timeout=10)


def test_send_flask_prediction_ok(flask_dummy_server, fake_input_dir, tmp_output_dir):
    wait_flask_alive(flask_dummy_server)
    send_flask_prediction(flask_dummy_server, 0, fake_input_dir, tmp_output_dir)


def test_collect_ok(flask_dummy_server):
    wait_flask_alive(flask_dummy_server)
    wait_flask_collect(flask_dummy_server, timeout=10)


def test_predict_dummy_server_raise(flask_dummy_server, fake_input_dir, tmp_output_dir):
    with pytest.raises(Exception):
        predict(flask_dummy_server, 0, fake_input_dir, tmp_output_dir, timeout=10)
