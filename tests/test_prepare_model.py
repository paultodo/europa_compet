import pytest
import os
import shutil
from prepare_model import prepare_model, does_model_exists, decompress_model_if_necessary
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Avoid gpu oom
from src.model import EuropaModel


@pytest.fixture
def fake_model_dir(tmpdir):
    "Create fake files to test model exist"
    for fname in ['model_fn.pkl', 'model_params.pkl',
                  'global_params.pkl', 'model_weights.h5']:
        open(os.path.join(str(tmpdir), fname), 'w').close()
    return str(tmpdir)


@pytest.fixture
def copy_model_dir(tmpdir, model_dir):
    tmp_model_dir = os.path.join(str(tmpdir), 'model/')
    shutil.copytree(
        model_dir,
        tmp_model_dir
    )
    return tmp_model_dir


@pytest.fixture
def ready_model_dir(copy_model_dir):
    decompress_model_if_necessary(copy_model_dir)
    return copy_model_dir


def test_does_model_exists(fake_model_dir):
    assert does_model_exists(fake_model_dir)


@pytest.mark.parametrize('missing_element', [
    'model_fn.pkl', 'model_params.pkl',
    'global_params.pkl', 'model_weights.h5'
])
def test_missing_element(fake_model_dir, missing_element):
    os.remove(os.path.join(fake_model_dir, missing_element))
    assert not does_model_exists(fake_model_dir)


def test_decompress_model(tmpdir, model_dir):
    shutil.copyfile(
        os.path.join(model_dir, 'model_weights.h5.xz'),
        os.path.join(str(tmpdir), 'model_weights.h5.xz')
    )
    decompress_model_if_necessary(str(tmpdir))
    assert os.path.isfile(os.path.join(str(tmpdir), 'model_weights.h5'))


def test_submission_model_dir(copy_model_dir):
    tmp_model_dir = copy_model_dir
    for fname in ['model_fn.pkl', 'model_params.pkl',
                  'global_params.pkl', 'model_weights.h5.xz']:
        assert os.path.isfile(
            os.path.join(tmp_model_dir, fname)
        )
    if os.path.isfile(os.path.join(tmp_model_dir, 'model_weights.h5')):
        os.remove(os.path.join(tmp_model_dir, 'model_weights.h5'))
    assert not does_model_exists(tmp_model_dir)
    decompress_model_if_necessary(tmp_model_dir)
    assert does_model_exists(tmp_model_dir)


def test_load_model(ready_model_dir):
    assert does_model_exists(ready_model_dir)
    model = EuropaModel.restore_model(ready_model_dir)
    assert isinstance(model, EuropaModel)


def test_prepare_model_from_solution(copy_model_dir, code_dir, fake_input_dir):
    configs = {
        'code_dir': code_dir,
        'input_dir': fake_input_dir,
        'model_dir': copy_model_dir,
        'HORIZON': 12,
    }
    prepare_model(configs)
    assert does_model_exists(copy_model_dir)


@pytest.mark.parametrize('horizon', [
    1, 12, 24,
])
def test_train_model(tmpdir, horizon, code_dir, fake_input_dir):
    model_dir = str(tmpdir)
    configs = {
        'code_dir': code_dir,
        'input_dir': fake_input_dir,
        'model_dir': model_dir,
        'HORIZON': horizon,
    }
    prepare_model(configs, test_run=True)
    assert does_model_exists(model_dir)
    model = EuropaModel.restore_model(model_dir)
    assert model.horizon == horizon
