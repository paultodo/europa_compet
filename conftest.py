import os
import pytest
import shutil


def pytest_addoption(parser):
    parser.addoption("--input_dir", action="store", required=True)
    parser.addoption("--output_dir", action="store", required=True)
    parser.addoption("--code_dir", action="store", required=True)


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".

    parameters = ['input_dir', 'output_dir', 'code_dir']
    for parameter in parameters:
        option_value = getattr(metafunc.config.option, parameter)
        if parameter in metafunc.fixturenames and option_value is not None:
            metafunc.parametrize(parameter, [option_value])


@pytest.fixture
def model_dir(code_dir):
    return os.path.join(code_dir, 'model/')


@pytest.fixture
def tmp_output_dir(tmpdir):
    return str(tmpdir)


def add_train_aux_files(aux_path, source):
    os.makedirs(os.path.join(aux_path, source), exist_ok=True)
    if source in ['opsd15', 'opsd60']:
        subsubdir = os.path.join(aux_path, source, 'Xm1')
        os.makedirs(subsubdir, exist_ok=True)
        open(os.path.join(subsubdir, 'X0.h5'), 'w').close()
    else:
        subsubdir = os.path.join(aux_path, source, 'X0')
        os.makedirs(subsubdir, exist_ok=True)
        if source == 'NOAA_NCOM_Region2':
            open(os.path.join(subsubdir, 'X0.tar.gz'), 'w').close()
        else:
            open(os.path.join(subsubdir, 'X0.h5'), 'w').close()


def add_adapt_aux_files(aux_path, source, step=0):
    os.makedirs(os.path.join(aux_path, source), exist_ok=True)
    if source in ['opsd15', 'opsd60']:
        subsubdir = os.path.join(aux_path, source, 'X{}'.format(step))
        os.makedirs(subsubdir, exist_ok=True)
        open(os.path.join(subsubdir, 'X0.h5'), 'w').close()
    else:
        subsubdir = os.path.join(aux_path, source, 'X{}'.format(step))
        os.makedirs(subsubdir, exist_ok=True)
        if source == 'NOAA_NCOM_Region2':
            open(os.path.join(subsubdir, 'X0.tar.nc.gz'), 'w').close()
        else:
            open(os.path.join(subsubdir, 'X0.h5'), 'w').close()


def add_fake_aux_files(train_or_adapt_path, is_train=True):
    aux_path = os.path.join(train_or_adapt_path, 'aux')
    os.makedirs(aux_path, exist_ok=True)
    for source in ['opsd15', 'opsd60', 'NOAA_NCOM_Region2', 'gosat_FTS_C01S_2']:
        if is_train:
            add_train_aux_files(aux_path, source)
        else:
            for step in range(10):
                add_adapt_aux_files(aux_path, source, step)


@pytest.fixture
def fake_input_dir(input_dir, tmpdir):
    tmp_input_dir = os.path.join(str(tmpdir), 'input/')
    shutil.copytree(input_dir, tmp_input_dir)
    assert os.path.isfile(os.path.join(tmp_input_dir, 'taskParameters.ini'))
    assert os.path.exists(os.path.join(tmp_input_dir, 'train'))
    add_fake_aux_files(os.path.join(tmp_input_dir, 'train'), is_train=True)
    assert os.path.exists(os.path.join(tmp_input_dir, 'adapt'))
    add_fake_aux_files(os.path.join(tmp_input_dir, 'adapt'), is_train=False)
    yield tmp_input_dir
    shutil.rmtree(tmp_input_dir)  # free up disk space
