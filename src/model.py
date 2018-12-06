import numpy as np
import os
import os.path
import shutil
import cloudpickle
import tensorflow as tf
import keras.layers as kl
from .generator import EuropaGenerator
from .generator_utils import create_2d_index, sample_time_index, get_batch_single_input
from .optimizers import NewAdam
from keras.callbacks import ModelCheckpoint, CSVLogger
from .multi_gpu_utils import multi_gpu_model


def quantize_weights(model, bound_quantize_percentile = 98, bits = 13):
    """
    Quantize the weights to take only 2**bits different values when they are in [- bound_quantize, bound_quantize].
    """
    weights = model.get_weights()

    line_weights = np.concatenate([e.ravel() for e in weights])
    line_weights.sort()

    lower_bound = np.percentile(line_weights, 100 - bound_quantize_percentile)
    upper_bound = np.percentile(line_weights, bound_quantize_percentile)

    line_weights = line_weights[(line_weights > lower_bound) & (line_weights < upper_bound)]
    line_weights = line_weights[::line_weights.shape[0] // (2**bits)]

    line_weights_mean = .5 * (line_weights[1:] + line_weights[:-1])

    new_weights = []
    for weight in weights:
        weight_indices = np.searchsorted(line_weights, weight) - 1
        weight_quantize = line_weights[np.clip(weight_indices.ravel(), 0, line_weights_mean.shape[0] - 1)]

        weight_indices = np.clip(weight_indices.ravel(), 0, line_weights.shape[0] - 1)
        weight_quantize = weight_quantize.reshape(weight.shape)

        weight_final = np.where((weight > lower_bound) & (weight < upper_bound), weight_quantize, weight)

        new_weights.append(weight_final)

    model.set_weights(new_weights)


def mse_timedistributed(y_true, y_pred):
    'MSE equivalent if predict is an ND tensor'
    loss = (y_true - y_pred)**2
    return tf.reduce_mean(loss)


class EuropaModel(object):
    """
    Utility class to manage keras model train, save, reload, predict cycle
    Typical cycle:
    model = EuropaModel(model_dir, model_fn, model_params)
    model.train(X_train, X_valid)
    ...
    model = EuropaModel.restore_model(model_dir)
    pred = model.predict(new_chunk_of_data)
    """
    def __init__(self, model_dir, model_fn, model_params,
                 hist_length=9216, horizon=12):
        """
        model_dir: where to save model
        model_fn, model_params: keral_model = model_fn(**model_params)
        """
        self.model_dir = model_dir
        self.model_fn = model_fn
        self.model_params = model_params
        self.horizon = horizon
        self.hist_length = hist_length
        self.keras_model = None
        self.is_built = False
        self.is_fitted = False

    def get_params(self):
        """Global params"""
        return dict(
            hist_length=self.hist_length,
            horizon=self.horizon
        )

    def build(self, gpus=[0]):
        "Build Keras model (not compiled)"
        if not self.is_built:
            if gpus == 0 or gpus == [0]:
                self.keras_model = self.model_fn(**self.model_params)
            else:
                self.keras_model = multi_gpu_model(
                    self.model_fn(**self.model_params),
                    gpus=gpus)
            self.is_built = True

    def train(self, X_train, X_valid=None,
              w_train=None,
              nb_epoch=12, batch_size=4000, time_step_per_epoch=4000,
              inverse_value_aug=False, inverse_time_aug=False,
              optimizer=NewAdam(1e-4, amsgrad=True), lr_callback=None,
              nb_worker=4, seed=1234, overwrite=False, gpus=[0]):
        """
        Train keras model
        X_train, X_valid: np.array
        w_train: weights for each lines in X_train
            array of shape (X_train.shape[0],)
        time_step_per_epoch: nb of time step sampled per epoch
            model will be trained on all lines at these time steps
        inverse_time_aug: add 1/2 chance to flip time order in train generator
        inverse_value_aug: add 1/2 chance to multiply values with -1 in train
            generator
        optimizer: optimizer to use
        lr_callback: callback to adjust learning rate schedule
            other callbacks during training are CSVLogger and ModelCheckpoint
        nb_worker: nb worker for generator
        seed: seed for generator
        overwrite: if True remove model_dir if exists when start training
        """
        if os.path.exists(self.model_dir):
            if overwrite:
                shutil.rmtree(self.model_dir, ignore_errors=True)
        os.makedirs(self.model_dir, exist_ok=True)
        train_gen = EuropaGenerator(X=X_train,
                                    w=w_train,
                                    hist_length=self.hist_length,
                                    output_horizon=self.horizon,
                                    batch_size=batch_size,
                                    time_step_per_epoch=time_step_per_epoch,
                                    resample_time_index=True,
                                    inverse_value_aug=inverse_value_aug,
                                    inverse_time_aug=inverse_time_aug,
                                    all_inputs=False,
                                    shuffle=True, seed=seed)
        if X_valid is not None:
            valid_time_index = sample_time_index(X_valid,
                                                 output_horizon=self.horizon,
                                                 min_sample_steps=12,
                                                 max_sample_steps=12*12)
            valid_gen = EuropaGenerator(X=X_valid,
                                        hist_length=self.hist_length,
                                        output_horizon=self.horizon,
                                        batch_size=batch_size,
                                        time_step_per_epoch=time_step_per_epoch,
                                        time_index=valid_time_index,
                                        resample_time_index=False,
                                        all_inputs=False,
                                        shuffle=False, seed=seed)
            nb_val_samples = valid_gen.N
        else:
            valid_gen = None
            nb_val_samples = None
        if not self.is_built:
            self.build(gpus=gpus)
        self.keras_model.compile(
            optimizer=optimizer,
            loss=mse_timedistributed)

        n_batchs_per_epoch = train_gen.nb_total_inputs * time_step_per_epoch / batch_size
        callbacks = [
            # ModelCheckpoint(self.model_dir + '/{epoch:02d}.h5'),
            CSVLogger(os.path.join(self.model_dir, 'train_log.csv')),
        ]
        if lr_callback is not None:
            callbacks.append(lr_callback)
        self.keras_model.fit_generator(
            train_gen, samples_per_epoch=n_batchs_per_epoch * batch_size,
            nb_epoch=nb_epoch,
            validation_data=valid_gen,
            nb_val_samples=nb_val_samples,
            callbacks=callbacks,
            max_q_size=20,
            nb_worker=nb_worker)
        self.is_fitted = True
        self.save()

    def predict(self, data):
        """
        Predict next points on given hitorical data
        data: numpy array of shape (hist_length, lines)
        """
        if not self.is_fitted:
            print('Model is not fitted. Predicting with random weights')
        if not self.is_built:
            self.build()
            self.keras_model.compile(
                optimizer='adam',
                loss=mse_timedistributed)
        
        try:
            assert np.sum(np.isnan(data)) == 0
        except:
            data = np.nan_to_num(data)
            
        try:
            assert len(data.shape) == 2
        except:
            if len(data.shape) == 1:
                data = data[:, np.newaxis]
            else:
                for _ in range(len(data.shape) - 2):
                    data = data[..., 0]
                    
        try:
            data = np.clip(data, a_min=-1e5, a_max=1e5)
        except:
            pass
            
        if len(data.shape) == 2:
            if data.shape[0] > self.hist_length:
                data = data[-self.hist_length:, :]
            elif data.shape[0] < self.hist_length:
                pad_length = self.hist_length - data.shape[0]
                data = np.pad(data, pad_width=((pad_length, 0), (0, 0)),
                              mode='constant', constant_values=0)
            x_b = data.T[:, :, np.newaxis]
            pred = self.keras_model.predict_on_batch(x_b)
            
            result = pred[:, :, 0].T
            try:
                assert np.sum(np.isnan(result)) == 0
            except:
                result = np.nan_to_num(result)
                
            try:
                result = np.clip(result, a_min=-1e5, a_max=1e5)
            except:
                pass
            return result
        else:
            raise ValueError('Inputs data for predict should have shape \
            (hist_length, lines)')

    def save(self):
        """
        Save elements to reload model object
        """
        os.makedirs(self.model_dir, exist_ok=True)
        with open(os.path.join(self.model_dir, 'model_fn.pkl'), 'wb') as f:
            cloudpickle.dump(self.model_fn, f)
        with open(os.path.join(self.model_dir, 'model_params.pkl'), 'wb') as f:
            cloudpickle.dump(self.model_params, f)
        with open(os.path.join(self.model_dir, 'global_params.pkl'), 'wb') as f:
            cloudpickle.dump(self.get_params(), f)
        if self.keras_model is not None:
            if isinstance(self.keras_model.layers[-1], kl.Merge):
                single_gpu_model = self.keras_model.layers[-2]
                quantize_weights(single_gpu_model)
                single_gpu_model.save_weights(
                    os.path.join(self.model_dir, 'model_weights.h5')
                )
            else:
                quantize_weights(self.keras_model)
                self.keras_model.save_weights(
                    os.path.join(self.model_dir, 'model_weights.h5')
                )

    @classmethod
    def restore_model(cls, model_dir, weights_path=None):
        """
        Reload model from folder for predict or continuous training
        """
        with open(os.path.join(model_dir, 'model_fn.pkl'), 'rb') as f:
            model_fn = cloudpickle.load(f)
        with open(os.path.join(model_dir, 'model_params.pkl'), 'rb') as f:
            model_params = cloudpickle.load(f)
        with open(os.path.join(model_dir, 'global_params.pkl'), 'rb') as f:
            global_params = cloudpickle.load(f)
        model = cls(model_dir=model_dir,
                    model_fn=model_fn,
                    model_params=model_params,
                    **global_params)
        model.build()
        if weights_path is None:
            weights_path = os.path.join(model_dir, 'model_weights.h5')
        if os.path.exists(weights_path):
            model.keras_model.load_weights(weights_path)
            model.is_fitted = True
        return model
