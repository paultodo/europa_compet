import numpy as np
from src.model import EuropaModel
from prepare_model import decompress_model_if_necessary
import pytest


@pytest.fixture
def loaded_model(model_dir):
    decompress_model_if_necessary
    model = EuropaModel.restore_model(model_dir)
    return model


def test_input_nan(loaded_model):
    in_ = np.empty(shape=(9000, 1916))
    in_[:] = np.nan
    pred = loaded_model.predict(in_)


def test_input_inf(loaded_model):
    in_ = np.empty(shape=(9000, 1916))
    in_[:] = np.inf
    pred = loaded_model.predict(in_)
