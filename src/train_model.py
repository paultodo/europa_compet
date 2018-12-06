import numpy as np
import os
import tensorflow as tf
from keras import layers as kl
from keras import models as km

from .model import EuropaModel
from .arch import multiscale_wavenet
from .callbacks import LrRestart
from .target_data_io import load_full_cache_matrixes, build_predict_matrix_from_cache_and_candidates_timestep


HIST_LEN = 64 * 12 * 12
BATCH_SIZE = 4500
N_GPUS = 8


def load_train_data(input_dir):
    # Load one year of history in train
    cache_X, cache_t = load_full_cache_matrixes(input_dir, -1, 12 * 24 * 365 + 100)

    predict_X = build_predict_matrix_from_cache_and_candidates_timestep(cache_X, cache_t, 12 * 24 * 365)
    return predict_X


def collapse_zeros_lines(X):
    # Collapse all zeros lines
    new_X_train = X[:, (X.var(axis=0) > 0).ravel()]
    new_X_train = np.hstack([new_X_train,
                             np.zeros(shape=(new_X_train.shape[0], 1))])
    weights_train = np.ones(shape=(new_X_train.shape[1],))
    weights_train[-1] = (X.var(axis=0) == 0).sum()
    return new_X_train, weights_train


def first_diff(x, padding=True):
    if padding:
        padded = tf.pad(x,
                        paddings=[[0, 0], [1, 0], [0, 0]],
                        mode="SYMMETRIC")
    else:
        padded = x
    diff = padded[:, 1:, :] - padded[:, :-1, :]
    return diff


def create_model(horizon, nb_inputs=1, input_length=HIST_LEN):
    raw_input = kl.Input(shape=(input_length, nb_inputs), name='X')

    # Compute diff
    diff = kl.Lambda(first_diff, arguments=dict(padding=True))(raw_input)
    out = multiscale_wavenet(input_length=input_length, nb_inputs=nb_inputs,
                             output_horizon=horizon,
                             nb_filters=128, batchnorm=False, bn_momentum=0.9,
                             hist_lengths=[64*12*12], time_units=[3*12], initial_subsamples=[3*12],
                             merge_scales='concat', intermediate_conv=False,
                             use_skip_connections=True, res_l2=0.0, final_l2=0.0,
                             dropout_rate=0.2, input_noise=0.)(diff)

    last_ = kl.Lambda(lambda x: x[:, -1:, :])(raw_input)
    pred = kl.Lambda(sum)([last_, out])

    model = km.Model(raw_input, pred)
    return model


def train_model(input_dir, model_dir, horizon,
                hist_length=HIST_LEN, batch_size=BATCH_SIZE,
                time_step_per_epoch=BATCH_SIZE, nb_epoch=18,
                **kwargs):
    X = load_train_data(input_dir)
    X_train, w = collapse_zeros_lines(X)

    model = EuropaModel(model_dir=model_dir,
                        model_fn=create_model,
                        model_params={'horizon': horizon},
                        hist_length=hist_length,
                        horizon=horizon)
    nb_steps_per_epoch = int(np.ceil(X_train.shape[1] *
                                     time_step_per_epoch /
                                     batch_size))
    lr_schedule = LrRestart(min_lr=5e-5, max_lr=5e-4,
                            cycle_length=nb_steps_per_epoch * nb_epoch / 2)

    model.train(X_train, X_valid=None, w_train=w,
                nb_epoch=nb_epoch,
                batch_size=batch_size,
                time_step_per_epoch=time_step_per_epoch,
                inverse_time_aug=False, inverse_value_aug=False,
                lr_callback=lr_schedule,
                gpus=[*range(N_GPUS)], nb_worker=4)
