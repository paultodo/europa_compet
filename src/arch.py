import numpy as np
import keras.layers as kl
import keras.models as km
import keras.backend as K
from keras.regularizers import l2
from .layers import SpatialDropout, mixture_experts


def persistence_model(hist_length, nb_inputs, output_horizon):
    """Model predicting always lastest value for all horizons
    hist_length : nb. of observations in history
    nb_inputs: nb. of input values per observation
    output_horizon: nb of horizons to predict in the future

    Input: (hist_length, nb_inputs)
    Ouput: (output_horizon, nb_inputs)
    """
    input_ = kl.Input(shape=(hist_length, nb_inputs))
    last_obs = kl.Lambda(lambda x: x[:, -1:, :])(input_)
    out = kl.Lambda(lambda x: K.repeat_elements(x, last_obs, axis=1))(last_obs)
    model = km.Model(input_, out)
    return model


def wavenet_block_light(x, nb_filters, subsample=2, use_bias=False,
                        res_l2=0., dropout_rate=0., batchnorm=False,
                        bn_momentum=0.99, **kwargs):
    """Conv block inspired by wavenet architecture.
    x : history of shape (batch_size, hist_length, nb_inputs, nb_features)
        x should be aranged in reverse time step, i.e latest obs is x[:,0,:,:]
    nb_filters : nb. of output features
    subsample : subsampling rate along time dimension
    use_bias : whether to use bias in conv layers
    res_l2 : l2 coef
    dropout_rate: spatial dropout rate
    batchnorm: use batchnorm if True
    bn_momentum: momentum coef for BatchNormalization
    """
    # TODO: Add padding in case time dimension not divisible by sub_sample
    dense = x

    if batchnorm:
        dense = kl.BatchNormalization(momentum=bn_momentum)(dense)
    dense = kl.Convolution2D(nb_filters,
                             nb_row=2, nb_col=1,
                             subsample=(subsample, 1),
                             border_mode='valid',
                             bias=use_bias, activation='relu',
                             W_regularizer=l2(res_l2))(dense)
    dense = SpatialDropout(dropout_rate, collapse_dim=(1,))(dense)
    res_x = kl.Convolution2D(nb_filters, nb_row=1, nb_col=1,
                             border_mode='same', bias=use_bias,
                             W_regularizer=l2(res_l2))(dense)

    subsampled_x = kl.Lambda(lambda x: x[:, 0::subsample, :, :])(x)
    res_out = kl.Merge(mode='sum')([subsampled_x, res_x])

    skip_out = kl.Lambda(lambda x: x[:, :1, :, :])(res_x)
    return res_out, skip_out


def wavenet_light(hist_length, nb_inputs, output_horizon,
                  nb_filters=128, dropout_rate=0., input_noise=0.,
                  nb_blocks=5, initial_pooling=1, initial_subsample=1,
                  use_skip_connections=True,
                  batchnorm=False, bn_momentum=0.99,
                  use_bias=False, res_l2=0., final_l2=0.,
                  has_top=True):
    """
    Simplified architecture inspired by wavenet.
    hist_length : nb. of observations in history
    nb_inputs: nb. of input values per observation
    output_horizon: nb of horizons to predict in the future
    has_top: if True return the prediction else return
        the last features layer
    input_tensor: use this at the input
    nb_blocks: nb of residual conv blocks
    initial_pooling: initial conv size
    initial_subsample: initial conv subsample
    use_skip_connections: if True sum over skip connections else use last conv
    dropout_rate: spatial dropout rate
    input_noise: add gaussian addtive noise to input
    batchnorm: use batchnorm if True
    bn_momentum: momentum coef for BatchNormalization
    """
    def arch(raw_input):
        head = raw_input

        if input_noise > 0.:
            head = kl.GaussianNoise(input_noise)(head)

        if initial_pooling > 1:
            head = kl.Lambda(
                lambda x: K.asymmetric_spatial_2d_padding(
                    x, top_pad=output_horizon, bottom_pad=0,
                    left_pad=0, right_pad=0))(head)
            if batchnorm:
                head = kl.BatchNormalization(momentum=bn_momentum)(head)
            head = kl.Convolution2D(nb_filters,
                                    nb_row=initial_pooling + output_horizon,
                                    nb_col=1, bias=use_bias,
                                    subsample=(initial_subsample, 1),
                                    border_mode='valid')(head)
        else:
            if batchnorm:
                head = kl.BatchNormalization(momentum=bn_momentum)(head)
            head = kl.Lambda(
                lambda x: K.asymmetric_spatial_2d_padding(
                    x, top_pad=1, bottom_pad=0,
                    left_pad=0, right_pad=0))(head)
            head = kl.Convolution2D(nb_filters,
                                    nb_row=2,
                                    nb_col=1, bias=use_bias,
                                    subsample=(1, 1),
                                    border_mode='valid')(head)

        perceptive_field = 2**nb_blocks * initial_subsample
        if perceptive_field < hist_length:
            print('History length of {} but conv block with perceptive field \
            of {}. This is suboptimal'.format(hist_length, perceptive_field))

        skip_connections = []
        for i in range(nb_blocks):
            head, skip_out = wavenet_block_light(head, nb_filters,
                                                 subsample=2,
                                                 use_bias=use_bias,
                                                 res_l2=res_l2,
                                                 batchnorm=batchnorm,
                                                 bn_momentum=bn_momentum,
                                                 dropout_rate=dropout_rate,
                                                 )
            skip_connections.append(skip_out)

        if use_skip_connections:
            head = kl.Merge(mode='sum')(skip_connections)
        else:
            head = kl.Lambda(lambda x: x[:, :1, :, :])(head)
        head = kl.Activation('relu')(head)

        if batchnorm:
            head = kl.BatchNormalization(momentum=bn_momentum)(head)
        head = kl.Convolution2D(nb_filters,
                                nb_row=1, nb_col=1,
                                border_mode='same', bias=use_bias,
                                )(head)
        head = kl.Dropout(dropout_rate)(head)

        if has_top:
            head = kl.Convolution2D(output_horizon,
                                    nb_row=1, nb_col=1,
                                    border_mode='same', bias=use_bias,
                                    W_regularizer=l2(final_l2))(head)
        return head
    return arch


def multiscale_wavenet(input_length, nb_inputs, output_horizon,
                       hist_lengths=[], time_units=[], initial_subsamples=[],
                       nb_filters=128,
                       merge_scales='concat', intermediate_conv=False,
                       use_skip_connections=True,
                       res_l2=0., final_l2=0.,
                       batchnorm=False, bn_momentum=0.99,
                       dropout_rate=0., input_noise=0., n_experts=1):
    """
    Architecture combining different wavenet-like blocks at different time scale
    input_length : nb. of observations in history
    nb_inputs: nb. of input values per observation
    output_horizon: nb of horizons to predict in the future

    hist_lengths: list. List of different length of history for each view
                  in nb of raw observations
    time_units: list. List of different conv size for each view
                in nb of raw observations
    initial_subsamples: list. List of different subsampling size for each view
                in nb of raw observations
    merge_scales: how to merge different scales output,
    intermediate_conv: if True add and intermediate layer before prediction
    use_skip_connections: if True sum over skip connections else use last conv
    n_experts: if >1, add mixture of experts
    """
    def arch(raw_input):
        # Reverse time dimension make it easier not to lose the lastest obs
        input_ = kl.Lambda(lambda x: x[:, ::-1, :])(raw_input)
        # Add a dimension for filters features -> shape (bs, time, inputs, features)
        input_ = kl.Lambda(lambda x: K.expand_dims(x))(input_)

        scale_outputs = []
        for hist_length, time_unit, initial_subsample in zip(hist_lengths, time_units, initial_subsamples):
            scale_input = kl.Lambda(lambda x: x[:, :hist_length, :, :])(input_)

            nb_blocks = int(np.log2(hist_length // initial_subsample))

            scale_out = wavenet_light(hist_length, nb_inputs, output_horizon,
                                      nb_filters=nb_filters,
                                      nb_blocks=nb_blocks,
                                      initial_pooling=time_unit,
                                      initial_subsample=initial_subsample,
                                      use_skip_connections=use_skip_connections,
                                      use_bias=False,
                                      res_l2=res_l2, final_l2=final_l2,
                                      batchnorm=batchnorm,
                                      bn_momentum=bn_momentum,
                                      dropout_rate=dropout_rate,
                                      input_noise=input_noise,
                                      has_top=False)(scale_input)
            scale_outputs.append(scale_out)
        if len(scale_outputs) > 1:
            out = kl.Merge(mode=merge_scales)(scale_outputs)
        else:
            out = scale_outputs[0]
        if intermediate_conv:
            if batchnorm:
                out = kl.BatchNormalization(momentum=bn_momentum)(out)
            out = kl.Convolution2D(nb_filters,
                                   nb_row=1, nb_col=1,
                                   border_mode='same', bias=False,
                                   )(out)
            out = kl.Dropout(dropout_rate)(out)

        if n_experts > 1:
            out = mixture_experts(out, output_horizon,
                                  final_l2=final_l2,
                                  n_experts=n_experts)
        else:
            out = kl.Convolution2D(output_horizon,
                                   nb_row=1, nb_col=1,
                                   border_mode='same',
                                   W_regularizer=l2(final_l2))(out)
        # Remove time dimension
        out = kl.Lambda(lambda x: K.squeeze(x, 1))(out)
        # Switch horizons into time dimension
        out = kl.Permute(dims=(2, 1))(out)
        return out
    return arch
