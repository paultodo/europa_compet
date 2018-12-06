import numpy as np
import threading


class Iterator(object):
    """
    Copy from keras.preprocessing.image
    """

    def __init__(self, N, batch_size, shuffle, seed):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    index_array = np.random.permutation(N)
                self.on_epoch_begin()

            current_index = (self.batch_index * batch_size) % N
            if N > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def on_epoch_begin(self):
        "Method inserted to add callbacks when a epoch start"
        return

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class EuropaGenerator(Iterator):

    def __init__(self, X, w=None,
                 hist_length=12, output_horizon=12, batch_size=32,
                 resample_time_index=False, time_step_per_epoch=4000,
                 time_index=None, inverse_value_aug=False, inverse_time_aug=False,
                 all_inputs=False, shuffle=False, seed=None):
        """
        X: np.float32, time series data
        w: weight for each lines in X, shape (X.shape[1],)
        hist_length: int, historical length to extract for each sample
        output_horizon: int, nb. of horizons to predict
        batch_size: int,
        resample_time_index: int, if True, resample time step after each epoch
        time_step_per_epoch: int, nb of time step to sample per epoch
        time_index: use predefined time index if not None
        all_inputs: boolean, if True return all inputs else one by one
        inverse_time_aug: add 1/2 chance to flip time order
        inverse_value_aug: add 1/2 chance to multiply values with -1
        """
        self.X = X

        if w is not None:
            assert len(w.shape) == 1
            assert w.shape[0] == X.shape[1]
        self.w = w

        self.hist_length = hist_length
        self.output_horizon = output_horizon

        self.time_step_per_epoch = time_step_per_epoch
        self.resample_time_index = resample_time_index

        self.inverse_value_aug = inverse_value_aug
        self.inverse_time_aug = inverse_time_aug

        self.all_inputs = all_inputs
        self.nb_total_inputs = X.shape[1]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.index = self._create_index(time_index)

        super(EuropaGenerator, self).__init__(len(self.index), batch_size,
                                              shuffle, seed)

        self._test_run()

    def _create_index(self, time_index=None):
        "Create index for iterator"
        if time_index is None:
            self.time_index = self._sample_time_index()
        else:
            self.time_index = time_index
        if self.all_inputs:
            index = self.time_index
        else:
            index = self._create_2d_index()
        return index

    def _sample_time_index(self):
        "Sample time step"
        if self.seed is None:
            seed = None
        else:
            try:
                seed = self.seed + self.total_batches_seen
            except:
                seed = self.seed
        total_steps = self.X.shape[0]
        time_index_groups = np.array_split(np.arange(total_steps),
                                           self.time_step_per_epoch)
        seeds = np.random.RandomState(seed).randint(0, 10000,
                                                    self.time_step_per_epoch)
        assert len(seeds) == len(time_index_groups)
        time_index = np.array([np.random.RandomState(s).choice(g)
                              for s, g in zip(seeds, time_index_groups)])
        assert len(time_index) == self.time_step_per_epoch
        return time_index

    def _create_2d_index(self):
        "combine time index and feature index to 2D index"
        feature_index = np.arange(self.nb_total_inputs)
        res = np.transpose([np.tile(self.time_index, len(feature_index)),
                            np.repeat(feature_index, len(self.time_index))])
        return res

    def on_epoch_begin(self):
        "Callback when epoc begin"
        if self.resample_time_index:
            self.index = self._create_index()

    def _extract_chunk(self, time_ind, feature_ind=None,
                       inverse_time=False, inverse_value=False):
        "Extract a chunk of time series"
        start_ = max(time_ind - self.hist_length, 0)
        end_ = time_ind + self.output_horizon
        if feature_ind is None:
            res = self.X[start_:end_, :]
            assert res.shape[1] == self.nb_total_inputs
        else:
            res = self.X[start_:end_, feature_ind]
            res = res[:, np.newaxis]
        assert len(res.shape) == 2

        if inverse_time:
            res = res[::-1, :]
        if inverse_value:
            res = -res
        return res

    def _pad_data(self, data):
        if len(data) < self.hist_length + self.output_horizon:
            pad_length = self.hist_length + self.output_horizon - len(data)
            data = np.pad(data, pad_width=((pad_length, 0), (0, 0)),
                          mode='constant', constant_values=0)
        assert len(data) == self.hist_length + self.output_horizon
        return data

    def _get_inputs_outputs(self, time_ind, feature_ind=None,
                            inverse_time=False, inverse_value=False):
        "Extract sample data given index"
        data = self._extract_chunk(time_ind, feature_ind,
                                   inverse_time=inverse_time,
                                   inverse_value=inverse_value)
        data = self._pad_data(data)
        x = data[:self.hist_length]
        y = data[-self.output_horizon:]
        return x, y

    def _get_batch(self, batch_index):
        "Extract batch data given index"
        x_b = list()
        y_b = list()

        if self.inverse_time_aug:
            inverse_time = np.random.uniform(0, 1, len(batch_index)) > 0.5
        else:
            inverse_time = np.zeros(shape=(len(batch_index)))
        if self.inverse_value_aug:
            inverse_value = np.random.uniform(0, 1, len(batch_index)) > 0.5
        else:
            inverse_value = np.zeros(shape=(len(batch_index)))
        for i, index in enumerate(batch_index.tolist()):
            if isinstance(index, int):
                x, y = self._get_inputs_outputs(time_ind=index,
                                                feature_ind=None,
                                                inverse_time=inverse_time[i],
                                                inverse_value=inverse_value[i],
                                                )
            elif isinstance(index, list):
                assert len(index) == 2
                x, y = self._get_inputs_outputs(time_ind=index[0],
                                                feature_ind=index[1],
                                                inverse_time=inverse_time[i],
                                                inverse_value=inverse_value[i],
                                                )
            x_b.append(x)
            y_b.append(y)

        x_b = np.stack(x_b)
        y_b = np.stack(y_b)

        if len(x_b.shape) == 2:
            x_b = x_b[:, :, np.newaxis]
        if len(y_b.shape) == 2:
            y_b = y_b[:, :, np.newaxis]

        if self.w is not None:
            w_b = []
            for index in batch_index.tolist():
                if isinstance(index, int):
                    w_b.append(1)
                elif isinstance(index, list):
                    assert len(index) == 2
                    feature_ind = index[1]
                    w_b.append(self.w[feature_ind])
            w_b = np.array(w_b).ravel()
        else:
            w_b = np.ones(shape=(len(batch_index),))
        return x_b, y_b, w_b

    def next(self, return_index=False):
        """Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = \
                next(self.index_generator)
        ind_ = self.index[index_array]
        batch_x, batch_y, batch_w = self._get_batch(ind_)
        if batch_w is None:
            batch_w = np.ones(len(index_array),)
        if return_index:
            index_array, batch_x, batch_y, batch_w
        return batch_x, batch_y, batch_w

    def _test_run(self):
        x_b, y_b, w_b = self.next()
        self.reset()
