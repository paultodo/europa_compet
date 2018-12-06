import os
import subprocess
import pytest
import requests
import filecmp
from src.flask_util import wait_flask_alive, send_flask_prediction, wait_flask_collect, predict
from src.utils import compact_trace
from main import start_prediction_server_function as start_prediction_server


def send_shutdown(server_full_address):
    try:
        answer = requests.get('{}/shutdown'.format(server_full_address)).json()
        assert answer['status'] == 'shutting down'
    except AssertionError:
        raise AssertionError('Expecting status "shutting down", instead got {}'.format(answer))
    except Exception as e:
        raise e


@pytest.fixture
def flask_prediction_server(code_dir, model_dir):
    try:
        send_shutdown('http://localhost:4130')
    except requests.exceptions.ConnectionError:
        pass
    except Exception as e:
        raise e
    configs = {'code_dir': code_dir, 'model_dir': model_dir}
    process = start_prediction_server(
        configs,
        gpu_count=0,
        port=4130,
        return_process=True
    )
    yield 'http://localhost:4130'
    try:
        send_shutdown('http://localhost:4130')
        process.terminate()
    except:
        pass


# Test utils
def test_wait_flask_alive(flask_prediction_server):
    wait_flask_alive(flask_prediction_server, timeout=10)


def test_wait_flask_alive_timeout():
    with pytest.raises(TimeoutError):
        wait_flask_alive('http://noserver', timeout=1)


def test_shutdown_request(flask_prediction_server):
    wait_flask_alive(flask_prediction_server)
    send_shutdown(flask_prediction_server)
    with pytest.raises(Exception):
        wait_flask_alive(flask_prediction_server, timeout=10)


def test_shutdown_correctly(flask_prediction_server):
    def dummy_test(flask_prediction_server):
        pass
    dummy_test(flask_prediction_server)
    with pytest.raises(Exception):
        wait_flask_alive('localhost:4130', timeout=10)


def test_send_flask_prediction(flask_prediction_server, fake_input_dir, tmp_output_dir):
    wait_flask_alive(flask_prediction_server)
    send_flask_prediction(flask_prediction_server, 0, fake_input_dir, tmp_output_dir)


def test_collect_no_server():
    with pytest.raises(requests.exceptions.ConnectionError):
        wait_flask_collect('http://noserver', timeout=10)


def test_collect_wrong_status(flask_prediction_server):
    wait_flask_alive(flask_prediction_server)
    with pytest.raises(TimeoutError):
        wait_flask_collect(flask_prediction_server, timeout=10)


def test_send_flask_prediction_no_server(fake_input_dir, tmp_output_dir):
    with pytest.raises(requests.exceptions.ConnectionError):
        send_flask_prediction('http://noserver', 0, fake_input_dir, tmp_output_dir)


@pytest.mark.parametrize('eval_step', [
    0, 1, 2, 50
])
def test_predict(flask_prediction_server, eval_step, fake_input_dir, tmp_output_dir):
    predict(flask_prediction_server, eval_step, fake_input_dir, tmp_output_dir)
    assert os.path.isfile(
        os.path.join(tmp_output_dir, 'Y{}.h5'.format(eval_step))
    )


def test_identical_predict(flask_prediction_server, fake_input_dir, tmp_output_dir, tmpdir):
    second_output_dir = str(tmpdir)
    for step in [0, 1, 50]:
        predict(flask_prediction_server, step, fake_input_dir, tmp_output_dir)
    for step in reversed([0, 1, 50]):
        predict(flask_prediction_server, step, fake_input_dir, second_output_dir)
    for step in [0, 1, 50]:
        assert filecmp.cmp(
            os.path.join(tmp_output_dir, 'Y{}.h5'.format(step)),
            os.path.join(second_output_dir, 'Y{}.h5'.format(step))
        )


def test_predict_wrong_input_dir(flask_prediction_server, tmp_output_dir, tmpdir):
    with pytest.raises(Exception):
        predict(flask_prediction_server, 0, str(tmpdir), tmp_output_dir)


def test_predict_output_dir_not_created(flask_prediction_server, fake_input_dir, tmp_output_dir):
    with pytest.raises(Exception):
        predict(flask_prediction_server, 0, fake_input_dir,
                os.path.join(tmp_output_dir, 'not_created_yet/'))


@pytest.fixture
def str_tmpdir(tmpdir):
    return str(tmpdir)


def test_launch_server_empty_model_dir(code_dir, fake_input_dir,
                                       tmp_output_dir, str_tmpdir):
    # Check no server is running
    try:
        send_shutdown('http://localhost:4130')
    except requests.exceptions.ConnectionError:
        pass
    except Exception as e:
        raise e
    # Launch server
    configs = {'code_dir': code_dir, 'model_dir': str_tmpdir}

    # Test
    with pytest.raises(Exception):
        try:
            process = start_prediction_server(
                configs,
                gpu_count=0,
                port=4130,
                return_process=True
            )

            predict('localhost:4130', 0, fake_input_dir, tmp_output_dir, timeout=30)
            assert os.path.isfile(
                os.path.join(tmp_output_dir, 'Y0.h5')
            )
        except Exception as e:
            print('Launch server empty model_dir. Raising exception {}'.format(e))
            raise e
    # Stop server
    try:
        send_shutdown('http://localhost:4130')
        process.terminate()
    except:
        pass


def test_launch_server_empty_code_dir(flask_prediction_server, code_dir, fake_input_dir, model_dir,
                                      tmp_output_dir, str_tmpdir):
    # Check no server is running
    try:
        send_shutdown('http://localhost:4130')
    except requests.exceptions.ConnectionError:
        pass
    except Exception as e:
        raise e
    # Launch server
    configs = {'code_dir': str_tmpdir, 'model_dir': model_dir}

    # Test
    with pytest.raises(Exception):
        try:
            process = start_prediction_server(
                configs,
                gpu_count=0,
                port=4130,
                return_process=True
            )

            predict('localhost:4130', 0, fake_input_dir, tmp_output_dir, timeout=30)
            assert os.path.isfile(
                os.path.join(tmp_output_dir, 'Y0.h5')
            )
        except Exception as e:
            print('Launch server empty model_dir. Raising exception {}'.format(e))
            raise e
    # Stop server
    try:
        send_shutdown('http://localhost:4130')
        process.terminate()
    except:
        pass

def test_compact_trace():
    out = compact_trace(['    traceback    ',
               '    text File "hello.py", line 26, other text    ',
               '    function name    ',
               '    ValueError    ',
               ''], max_row_size=20)

    assert out == 'hello.py : 26,\nValueError'

    out = compact_trace(['    traceback    ',
                   '    text File "hello.py", line 26, other text    ',
                   '    function print_hello world    ',
                   '    some text File "world.py", line 91, some other text    ',
                   '    some function name    ',
                   '    ValueError    ',
                   ''], max_row_size=20)

    assert out == 'world.py : 91,\nhello.py : 26,\nValueError'

    out = compact_trace(['    traceback    ',
                   '    text File "some/complicated/path/hello.py", line 26, other text    ',
                   '    function print_hello world    ',
                   '    some text File "some/very/complicated/path/world.py", line 91, some other text    ',
                   '    some function name    ',
                   '    ValueError    ',
                   ''], max_row_size=20)

    assert out == 'ld.py : 91, complica\nted/path/hello.py :\n26,  ValueError'

    out = compact_trace(['error in parsing 1',
                         'error in parsing 2',
                         'error in parsing 3',
                         'error in parsing 4'], max_row_size=40)

    assert out == 'error in parsing 1 error in parsing 2\nerror in parsing 3 error in parsing 4'
