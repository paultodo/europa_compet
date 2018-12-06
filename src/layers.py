import tensorflow as tf
import keras.backend as K
from keras.layers import Layer, AtrousConvolution1D, AtrousConvolution2D, \
    Permute, Lambda, Convolution2D, Merge
from keras.regularizers import l2
from keras.utils.np_utils import conv_output_length


def categorical_mean_squared_error(y_true, y_pred):
    """MSE for categorical variables."""
    return K.mean(K.square(K.argmax(y_true, axis=-1) -
                           K.argmax(y_pred, axis=-1)))


class CausalAtrousConvolution1D(AtrousConvolution1D):
    def __init__(self, nb_filter, filter_length, init='glorot_uniform', activation=None, weights=None,
                 border_mode='valid', subsample_length=1, atrous_rate=1, W_regularizer=None, b_regularizer=None,
                 activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, causal=False, **kwargs):
        super(CausalAtrousConvolution1D, self).__init__(nb_filter, filter_length, init, activation, weights,
                                                        border_mode, subsample_length, atrous_rate, W_regularizer,
                                                        b_regularizer, activity_regularizer, W_constraint, b_constraint,
                                                        bias, **kwargs)
        self.causal = causal
        if self.causal and border_mode != 'valid':
            raise ValueError("Causal mode dictates border_mode=valid.")

    def get_output_shape_for(self, input_shape):
        input_length = input_shape[1]

        if self.causal:
            input_length += self.atrous_rate * (self.filter_length - 1)

        length = conv_output_length(input_length,
                                    self.filter_length,
                                    self.border_mode,
                                    self.subsample[0],
                                    dilation=self.atrous_rate)

        return (input_shape[0], length, self.nb_filter)

    def call(self, x, mask=None):
        if self.causal:
            x = K.asymmetric_temporal_padding(x, left_pad=self.atrous_rate * (self.filter_length - 1), right_pad=0)
        return super(CausalAtrousConvolution1D, self).call(x, mask)


class CausalAtrousConvolution2D(AtrousConvolution2D):
    """Causal: dim row represent time step, dim col represent entities (power line)"""
    def __init__(self, nb_filter, nb_row, nb_col, init='glorot_uniform', activation=None, weights=None,
                 border_mode='valid', subsample=(1, 1), atrous_rate=(1, 1), dim_ordering='default',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, causal=False, **kwargs):
        super(CausalAtrousConvolution2D, self).__init__(nb_filter, nb_row, nb_col, init, activation, weights,
                                                        border_mode, subsample, atrous_rate, dim_ordering, W_regularizer,
                                                        b_regularizer, activity_regularizer, W_constraint, b_constraint,
                                                        bias, **kwargs)
        self.causal = causal
        if self.causal and border_mode != 'valid':
            raise ValueError("Causal mode dictates border_mode=valid.")

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)
        if self.causal:
            rows += self.atrous_rate[0] * (self.nb_row - 1)

        rows = conv_output_length(rows, self.nb_row, self.border_mode,
                                  self.subsample[0],
                                  dilation=self.atrous_rate[0])
        cols = conv_output_length(cols, self.nb_col, self.border_mode,
                                  self.subsample[1],
                                  dilation=self.atrous_rate[1])
        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)

    def call(self, x, mask=None):
        if self.causal:
            x = K.asymmetric_spatial_2d_padding(
                x,
                top_pad=self.atrous_rate[0] * (self.nb_row - 1),
                bottom_pad=0, left_pad=0, right_pad=0)
        return super(CausalAtrousConvolution2D, self).call(x, mask)


class SliceTimeDimension(Layer):
    """DEPRECATED: somehow keras does not recognize correctly the ouput shape"""
    def __init__(self, start, stop, step):
        self.slice = slice(start, stop, step)
        super(SliceTimeDimension, self).__init__()

    def build(self, input_shape):
        super(SliceTimeDimension, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if len(input_shape < 3) or len(input_shape > 4):
            raise ValueError('Input of {} must be 3D or 4D')
        all_slices = [i for i in range(input_shape[1])]
        n_output_slices = len(all_slices[self.slice])
        return tuple(input_shape[0], n_output_slices, *input_shape[2:])

    def call(self, x, mask=None):
        if len(x.shape) == 3:
            return x[:, self.slice, :]
        elif len(x.shape) == 4:
            return x[:, self.slice, :, :]
        else:
            raise NotImplementedError('Expect input to have dimensions 3 or 4. \
            Instead got shape : {}'.format(x.shape))


class SpatialDropout(Layer):
    """Applies Dropout to the input.

    Dropout consists in randomly setting
    a fraction `p` of input units to 0 at each update during training time,
    which helps prevent overfitting.

    # Arguments
        p: float between 0 and 1. Fraction of the input units to drop.
        collapse_dim: tuple or list of integer representing dimensions to
            have the same droppout value across values
        seed: A Python integer to use as random seed.

    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    """

    def __init__(self, p, seed=None, collapse_dim=(), **kwargs):
        self.p = p
        self.seed = seed
        if 0. < self.p < 1.:
            self.uses_learning_phase = True
        self.supports_masking = True
        self.collapse_dim = collapse_dim
        self.ndims = None
        super(SpatialDropout, self).__init__(**kwargs)

    def _get_noise_shape(self, _):
        return self.noise_shape

    def build(self, input_shape):
        self.ndims = len(input_shape)
        super(SpatialDropout, self).build(input_shape)

    def call(self, x, mask=None):
        if 0. < self.p < 1.:
            input_shape = K.shape(x)
            noise_shape = []
            for i in range(self.ndims):
                if i in self.collapse_dim:
                    noise_shape.append(1)
                else:
                    noise_shape.append(input_shape[i])
            def dropped_inputs():
                return K.dropout(x, self.p, noise_shape, seed=self.seed)
            x = K.in_train_phase(dropped_inputs, lambda: x)
        return x

    def get_config(self):
        config = {'p': self.p}
        base_config = super(SpatialDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def mixture_experts(input_, output_horizon, final_l2=0., n_experts=2):
    mix = Convolution2D(n_experts,
                        nb_row=1, nb_col=1,
                        border_mode='same')(input_)
    mix = Lambda(lambda x: tf.nn.softmax(x, dim=-1))(mix)
    mix = Lambda(lambda x: K.repeat_elements(x, output_horizon, axis=1))(mix)
    mix = Permute((3, 2, 1))(mix)


    experts = [Convolution2D(output_horizon,
                             nb_row=1, nb_col=1,
                             border_mode='same',
                             W_regularizer=l2(final_l2))(input_)
               for _ in range(n_experts)]
    experts = Merge(mode='concat', concat_axis=1)(experts)

    out = Merge(mode='mul')([mix, experts])
    out = Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(out)
    return out
