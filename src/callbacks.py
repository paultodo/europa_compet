import numpy as np
import keras.backend as K
from keras.callbacks import Callback


class LrRestart(Callback):
    """
    SGD with restart
    """
    def __init__(self, min_lr=0.001, max_lr=0.006,
                 cycle_length=2000., mult=1, add=0,
                 keep_logs=False):
        super(LrRestart, self).__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length
        self.mult = mult
        self.add = add

        self.cycle_iterations = 0.
        self.total_iterations = 0.
        self.history = {}
        self.keep_logs = keep_logs

    def lr(self):
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
            (1 + np.cos(self.cycle_iterations / self.cycle_length * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        logs = logs or {}
        if self.total_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.min_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.lr())

    def on_batch_end(self, batch, logs=None):
        self.cycle_iterations += 1
        self.total_iterations += 1
        if self.cycle_iterations == self.cycle_length:
            self.cycle_length = self.cycle_length * self.mult + self.add
            self.cycle_iterations = 0
        if self.keep_logs:
            logs = logs or {}
            self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
            self.history.setdefault('iterations', []).append(self.total_iterations)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)
        K.set_value(self.model.optimizer.lr, self.lr())
