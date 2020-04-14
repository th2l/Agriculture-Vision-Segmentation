from comet_ml import Experiment
from comet_ml.exceptions import InterruptedExperiment

import os, sys, random, glob
import numpy as np
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import tensorflow as tf

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# tf.compat.v1.disable_eager_execution()
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

from scipy.ndimage import distance_transform_edt as distance
import pathlib
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils import tf_utils
from tqdm import tqdm
from PIL import Image
import time
import tabulate
from functools import partial
if os.name != 'nt':
    import tensorflow_addons as tfa

label_dict = {0: 'background', 1: 'cloud_shadow', 2: 'double_plant', 3: 'planter_skip', 4: 'standing_water',
              5: 'waterway', 6: 'weed_cluster'}

# tf.get_logger().setLevel('INFO')
mean = tf.constant([0.42232055, 0.45105023, 0.46550912, 0.47275587, ], dtype=tf.float32)  # Included invalid pixels, ndvi = -0.03749848
std = tf.constant([0.17638557, 0.1783236, 0.16594722, 0.16398345,], dtype=tf.float32)   # Included invalid pixels ndvi =  0.25743752
# mean = tf.constant([0.35581957, 0.40761307, 0.41166625, 0.42395257], dtype=tf.float32)  # Only valid pixels (zero out invalid pixels), ndvi = -0.06318427
# std = tf.constant([0.19890528, 0.21811342, 0.21531802, 0.21479421], dtype=tf.float32)   # Only valid pixels (zero out invalid pixels)
AUTOTUNE = tf.data.experimental.AUTOTUNE
initializer =  'he_normal' #'glorot_uniform' # 'glorot_normal'  # #, , # , 'he_uniform'

"""
Model define functions
"""
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def squeeze_excitation_block(inputs, reduction=1, add_conv=False):
    """Squeeze and Excitation.
    This function defines a squeeze structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
    """

    input_channels = int(inputs.shape[-1])

    if add_conv:
        x = ConvBNReLU(inputs, filters=input_channels, kn_size=3)
    else:
        x = inputs
    x = GlobalAveragePooling2D()(x)
    x = Dense(input_channels // reduction, activation='relu', kernel_initializer=initializer)(x)
    x = Dense(input_channels, activation='hard_sigmoid', kernel_initializer=initializer)(x)
    x = Reshape((1, 1, input_channels))(x)
    x = tf.keras.layers.Multiply()([inputs, x])

    return x

def hard_swish(self, x):
    """Hard swish
    """
    return x * tf.nn.relu6(x + 3.0) / 6.0

def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, kn_size=3, se_connect=False, activation='relu6'):
    in_channels = inputs.shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    if activation == 'hswish':
        act_func = hard_swish
    else:
        act_func = tf.nn.relu6

    if block_id:
        # Expand
        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None, kernel_initializer=initializer,
                   name='mobl%d_conv_expand' % block_id)(inputs)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name='bn%d_conv_bn_expand' %
                                    block_id)(x)
        x = Activation(act_func, name='block_%d_expand_relu' % block_id)(x)
    else:
        x = inputs

    # Depthwise
    x = DepthwiseConv2D(kernel_size=kn_size, strides=stride, activation=None,
                        use_bias=False, padding='same', kernel_initializer=initializer,
                        name='mobl%d_conv_depthwise' % block_id)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name='bn%d_conv_depthwise' % block_id)(x)

    x = Activation(tf.nn.relu6, name='conv_dw_%d_relu' % block_id)(x) if activation == 'relu6' else hard_swish(x)

    if se_connect:
        x = squeeze_excitation_block(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,kernel_initializer=initializer,
               name='mobl%d_conv_project' % block_id)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name='bn%d_conv_bn_project' % block_id)(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x

def SepConv_BN(x, filters, prefix="aspp", stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3, momentum=0.999):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
        
        https://github.com/bonlime/keras-deeplab-v3-plus
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation(tf.nn.relu)(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon, momentum=momentum)(x)
    if depth_activation:
        x = Activation(tf.nn.relu)(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon, momentum=momentum)(x)
    if depth_activation:
        x = Activation(tf.nn.relu)(x)

    return x

def AgriMobileNetV2(input_shape=None,
                    alpha=1.0,
                    classes=7, n_channels=4):
    """Instantiates the MobileNetV2 architecture.
    To load a MobileNetV2 model via `load_model`, import the custom
    objects `relu6` and pass them to the `custom_objects` parameter.
    E.g.
    model = load_model('mobilenet.h5', custom_objects={
                       'relu6': mobilenet.relu6})
    # Arguments
        input_shape: optional shape tuple, to be specified if you would
            like to use a model with an input img resolution that is not
            (224, 224, 3).
        alpha: controls the width of the network. This is known as the
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    """
    # Determine proper input shape and default size.
    # If both input_shape and input_tensor are used, they should match
    row_axis, col_axis = (0, 1)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    img_input = Input(shape=input_shape)

    first_block_filters = _make_divisible(32 * alpha, 8)

    x = squeeze_excitation_block(img_input, reduction=1, add_conv=True)
    # x = img_input
    x = Conv2D(first_block_filters,
               kernel_size=3,
               strides=(2, 2), padding='same',kernel_initializer=initializer,
               use_bias=False, name='Conv1')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
    x = Activation(tf.nn.relu6, name='Conv1_relu')(x)

    inverted_residual_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
    ]

    block_id = 0
    for t, c, n, s in inverted_residual_setting:
        for i in range(n):
            stride = s if i == 0 else 1
            x = _inverted_res_block(x, filters=c, alpha=alpha, stride=stride, expansion=t, block_id=block_id)
            block_id += 1

    last_block_filters = 64

    x = Conv2D(last_block_filters, kernel_size=1, use_bias=False, name='Conv_1',kernel_initializer=initializer)(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
    x = Activation(tf.nn.relu6, name='block_16_project')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    inputs = img_input

    # Create model.
    model = Model(inputs, x, name='AgriMobilenetv2_%0.2f_%s_%s' % (alpha, rows, cols))

    return model

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False, upsample_method='conv', up_size=4):
    """Upsamples an input.
    Conv2DTranspose => Batchnorm => Dropout => Relu
    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
      apply_dropout: If True, adds the dropout layer
    Returns:
      Upsample Sequential Model
    """
    result = tf.keras.Sequential()
    if upsample_method == 'conv':
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=up_size,
                                            padding='same',kernel_initializer=initializer,
                                            use_bias=False))
        # result.add(
        #     tf.keras.layers.Conv2D(filters, size, padding='same', use_bias=False))
    else:
        result.add(
            tf.keras.layers.Conv2D(filters, size, padding='same', use_bias=False, kernel_initializer=initializer))

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform"))

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    if upsample_method == 'bilinear' or upsample_method == 'nearest':
        result.add(tf.keras.layers.UpSampling2D(size=up_size, interpolation=upsample_method))
    return result


def ConvBNReLU(inp, filters=64, kn_size=3):
    x = tf.keras.layers.Conv2D(
            filters, kn_size, strides=1,kernel_initializer=initializer,
            padding='same', use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = Activation(tf.nn.relu)(x)
    return x

def up_last(tensor, kn_size=3, out_channels=7, up_size=2, n_conv=3):
    x = ConvBNReLU(tensor, out_channels, kn_size)

    for ncov in range(1, n_conv):
        x = ConvBNReLU(x, out_channels, kn_size)

    x = tf.keras.layers.UpSampling2D(size=up_size, interpolation="bilinear")(x)
    return x

def up_last_SE(tensor, kn_size=3, out_channels=7, up_size=2):
    x = ConvBNReLU(tensor, out_channels, kn_size)
    x = tf.keras.layers.UpSampling2D(size=up_size, interpolation="bilinear")(x)

    squeeze_excitation_block(x, reduction=1)

    return x

def AgriVi_Segmentation(upsample_method='conv', output_channels=7, unet=False):
    if upsample_method == 'nearest':
        raise ValueError("{} not support DETERMINISTIC as this time".format(upsample_method))
    n_channels = 4
    base_model = AgriMobileNetV2(input_shape=(512, 512, n_channels), classes=n_classes)
    layer_names = ['block_3_expand_relu', 'block_6_expand_relu', 'block_16_project']
    # layer_names = ['block_2_expand_relu', 'block_5_expand_relu', 'block_16_project']

    layers = [base_model.get_layer(name).output for name in layer_names]
    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    # print("Encoder summary:\n", down_stack.summary())

    inputs = tf.keras.layers.Input(shape=[512, 512, n_channels])
    x = inputs

    # Down-sampling through the encoder model
    encoder = down_stack(x)
    x = encoder[-1]
    # x = Dropout(0.2)(x)

    # Up-sampling with Conv2D/bilinear/nearest
    upsample_method = upsample_method

    n_filters = 64
    xenc_f_shape = x.shape[1:3]
    xenc_f_pooling = GlobalAveragePooling2D()(x)  # ENcoder index  x, 
    xenc_f_pooling = tf.expand_dims(tf.expand_dims(xenc_f_pooling, 1), 1)
    xenc_f_pooling = ConvBNReLU(xenc_f_pooling, filters=n_filters, kn_size=1)
    xenc_f_pooling = tf.keras.layers.UpSampling2D(size=xenc_f_shape, interpolation="bilinear")(xenc_f_pooling)
    
    xenc_f_conv1 = ConvBNReLU(x, filters=n_filters, kn_size=1)
    atrous_rates = [6, 12, 18]
    xenc_f_conv31 = SepConv_BN(x, n_filters, 'aspp1',
                        rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    xenc_f_conv32 = SepConv_BN(x, n_filters, 'aspp2',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    xenc_f_conv33 = SepConv_BN(x, n_filters, 'aspp3',
                        rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

    xenc_f_cc = tf.keras.layers.Concatenate()([xenc_f_pooling, xenc_f_conv1, xenc_f_conv31, xenc_f_conv32, xenc_f_conv33])
    # xenc_f_cc = squeeze_excitation_block(xenc_f_cc, reduction=16)
    x = ConvBNReLU(xenc_f_cc, n_filters, kn_size=1)
    x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)

    xenc_2 = encoder[-2] #squeeze_excitation_block(encoder[-2], reduction=16)
    xenc_2 = ConvBNReLU(xenc_2, n_filters // 2, kn_size=1)
    x = tf.keras.layers.Concatenate()([x, xenc_2])
    x = ConvBNReLU(x, n_filters, kn_size=3)
    x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)

    xenc_3 = encoder[-3] # squeeze_excitation_block(encoder[-3], reduction=16)
    xenc_3 = ConvBNReLU(xenc_3, n_filters // 4, kn_size=3)
    x = tf.keras.layers.Concatenate()([x, xenc_3])

    x_last = up_last(x, kn_size=3, out_channels=7, up_size=4, n_conv=3)
    x_last_SE = up_last_SE(x, kn_size=3, out_channels=7, up_size=4)

    x = x_last  + x_last_SE
    
    return tf.keras.Model(inputs=inputs, outputs=x)

"""
Loss,  Metric, Callback functions
"""
class AgrVimIOU(tf.keras.metrics.Metric):
    def __init__(self, name='AgrVimIOU', num_classes=7, per_classes=False):
        super(AgrVimIOU, self).__init__(name=name)
        self.num_classes = num_classes
        self.arr_mult = tf.range(1, self.num_classes + 1, dtype=tf.float32)

        self.total_true_positive = self.add_weight(
            'total_true_positive',
            shape=(self.num_classes,),
            initializer=tf.zeros_initializer,
            dtype=tf.float64)

        self.total_union = self.add_weight(
            'total_union',
            shape=(self.num_classes,),
            initializer=tf.zeros_initializer,
            dtype=tf.float64)
        self.per_classes = per_classes

    def update_state(self, y_true, y_pred, sample_weight=None):
        """

        :param y_true: batch_size x 512 x 512 x 8
        :param y_pred: batch_size x 512 x 512
        :param sample_weight:
        :return:
        """
        invalid_pixels = tf.expand_dims(tf.cast(y_true[:, :, :, -1], tf.bool), -1)  # Index of invalid pixels

        y_true_onehot = tf.cast(y_true[:, :, :, :-1], tf.float32)
        
        y_pred_onehot = tf.one_hot(tf.argmax(tf.nn.softmax(y_pred, axis=-1), axis=-1), depth=self.num_classes, axis=-1, dtype=tf.float32)
        # y_pred_onehot = tf.cast(tf.hard_sigmoid(y_pred) > 0.5, dtype=tf.float32)

        overlap = tf.reduce_sum(tf.multiply(y_true_onehot, y_pred_onehot), axis=-1, keepdims=True)

        y_pred_onehot = tf.where(tf.logical_or(tf.equal(overlap, 1), invalid_pixels), y_true_onehot, y_pred_onehot)
        # y_pred_onehot = tf.where(invalid_pixels, tf.zeros_like(y_pred_onehot), y_pred_onehot)

        current_true_positive = tf.cast(tf.reduce_sum(tf.multiply(y_pred_onehot, y_true_onehot), axis=[0, 1, 2]), tf.float64)
        current_union = tf.cast(tf.reduce_sum(y_pred_onehot + y_true_onehot, axis=[0, 1, 2]), tf.float64) - current_true_positive

        self.total_true_positive.assign_add(current_true_positive)
        self.total_union.assign_add(current_union)
        

    def result(self):
        """Compute the mean intersection-over-union via the confusion matrix."""

        num_valid_entries = tf.cast(tf.reduce_sum(tf.cast(tf.not_equal(self.total_union, 0), tf.int32)), tf.float64)

        iou = tf.math.divide_no_nan(self.total_true_positive, self.total_union)

        if self.per_classes:
            tf.print(iou, summarize=-1)
        return tf.math.divide_no_nan(tf.reduce_sum(iou, name='agrvi_mean_iou'), num_valid_entries)

    def reset_states(self):
        tf.keras.backend.set_value(self.total_true_positive, np.zeros((self.num_classes)))
        tf.keras.backend.set_value(self.total_union, np.zeros((self.num_classes)))

    def get_config(self):
        return {'name': self.name, 'num_classes': self.num_classes}

    @classmethod
    def from_config(cls, config):
        """Instantiates a `Loss` from its config (output of `get_config()`).
        Args:
            config: Output of `get_config()`.
        Returns:
            A `Loss` instance.
        """
        return cls(**config)

    
class AgrViCBLosses(object):
    
    def __init__(self, name='AgrVi_loss_CB', num_classes=7, beta=0.9999, gamma=0.5, sf_coeff=0.0):
        self.name = name
        self.num_classes = num_classes
        self.gamma = gamma
        self.beta = beta
        self.sf_coeff = sf_coeff

    def onehot_distmaps(self, y_true):
        """
        Surface loss from https://github.com/LIVIAETS/surface-loss
        Convert y_true to distance map 
        :param y_true: batch_size x 512 x 512 x 8
        """
        y_true_ = y_true[:, :, :, :-1]
        invalid_ = tf.expand_dims(tf.cast(y_true[:, :, :, -1], tf.bool), axis=-1)
        
        ld_dist = [None, None, None, None, None, None, None]
        for cidx in range(self.num_classes):
            cr_y = tf.expand_dims(tf.cast(y_true_[:, :, :, cidx], tf.uint8), axis=-1)
            cr_y_neg = tf.cast(1-cr_y, tf.uint8)
            # cr_y_neg = tf.where(invalid_, tf.zeros_like(cr_y_neg), cr_y_neg)           

            res = tf.multiply(tfa.image.euclidean_dist_transform(cr_y_neg), tf.cast(cr_y_neg, tf.float32)) - tf.multiply(tfa.image.euclidean_dist_transform(cr_y) - 1, tf.cast(cr_y, tf.float32))
            cf = tf.where(tf.reduce_sum(y_true_[:, :, :, cidx]) > 0, 1.0, 0.0)

            res = tf.where(invalid_, tf.zeros_like(res), res)
            ld_dist[cidx] = cf*res

        ret = tf.concat(ld_dist, axis=-1)
        
        return ret

    def __call__(self, y_true, y_pred, sample_weight=None):
        """
        https://github.com/richardaecn/class-balanced-loss
        :param y_true: batch_size x 512 x 512 x 8
        :param y_pred: batch_size x 512 x 512 x 7
        :param sample_weight:
        :return:
        """
        scope_name = 'lambda' if self.name == '<lambda>' else self.name
        graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
            y_true, y_pred, sample_weight)
        with tf.name_scope(scope_name or self.__class__.__name__), graph_ctx:
            invalid_pixels = tf.expand_dims(tf.cast(y_true[:, :, :, -1], tf.bool), -1)  # Index of invalid pixels
            n_valid_pixels = tf.size(invalid_pixels) - tf.reduce_sum(tf.cast(invalid_pixels, tf.int32))
            mean_factor = tf.cast(tf.size(invalid_pixels), tf.float32) / tf.cast(n_valid_pixels, tf.float32)

            # Calculate loss value
            # y_true_mask = tf.cast(y_true[:, :, :, :-1], tf.float32) / tf.cast(tf.reduce_sum(y_true, axis=-1, keepdims=True), tf.float32)
            y_true_mask = tf.cast(y_true[:, :, :, :-1], tf.float32)

            red_dim = np.arange(0, y_true_mask.shape.ndims - 1, 1).tolist()
            # ref_vol = tf.reduce_sum(tf.cast(y_true[:, :, :, :-1], tf.float32), axis=[1, 2])   # Out: batch_size x n_classes
            # n_valid_class = tf.math.count_nonzero(ref_vol, dtype=tf.float32)


            # effective_num = 1.0 - tf.math.pow(self.beta, ref_vol)       # Out: batch_size x n_classes
            # weight = (1.0 - self.beta) * tf.math.reciprocal(effective_num)   # Out: batch_size x n_classes

            # weight = tf.where(tf.math.is_inf(weight), tf.zeros_like(weight), weight)    # Out: batch_size x n_classes
            # alpha = weight / tf.reduce_sum(weight, axis=-1, keepdims=True) * n_valid_class            # Out: batch_size x n_classes
            # alpha = tf.expand_dims(tf.expand_dims(alpha, axis=1), axis=1)
            # alpha = tf.multiply(alpha, y_true_mask)
            
            ref_vol = tf.reduce_sum(tf.cast(y_true[:, :, :, :-1], tf.float32), axis=[0, 1, 2])   # Out: n_classes  axis=[0, 1, 2]
            n_valid_class = tf.math.count_nonzero(ref_vol, dtype=tf.float32)
            
            # tf.print("DEBUG: ", n_valid_class == 7)

            effective_num = 1.0 - tf.math.pow(self.beta, ref_vol)       # Out: n_classes
            weight = (1.0 - self.beta) * tf.math.reciprocal(effective_num)   # Out: n_classes

            weight = tf.where(tf.math.is_inf(weight), tf.zeros_like(weight), weight)    # Out: n_classes
            alpha = weight / tf.reduce_sum(weight) * n_valid_class    # Out: n_classes
            alpha = tf.expand_dims(tf.expand_dims(tf.expand_dims(alpha, axis=0), axis=0), axis=0)  # Out: 1 x 1 x 1 x n_classes
            # alpha = tf.expand_dims(tf.expand_dims(alpha, axis=1), axis=1)  # Out: 1 x 1 x 1 x n_classes
            alpha = tf.multiply(alpha, y_true_mask)
            
            # cross_entropy = tf.nn.hard_sigmoid_cross_entropy_with_logits(labels=y_true_mask, logits=y_pred)   # hard_sigmoid_cross_entropy

            y_true_mask = y_true_mask / tf.cast(tf.reduce_sum(y_true, axis=-1, keepdims=True), tf.float32)
            y_pred_mask = tf.nn.softmax(y_pred, axis=-1)
            cross_entropy = -tf.multiply(y_true_mask, tf.math.log(y_pred_mask))

            # A numerically stable implementation of modulator.
            if self.gamma == 0.0:
                modulator = 1.0
            else:
                modulator = tf.exp(-self.gamma * y_true_mask * y_pred_mask - self.gamma * tf.math.log1p(tf.math.exp(-1.0 * y_pred_mask)))

            weighted_loss = tf.reduce_sum(tf.multiply(alpha, modulator * cross_entropy), axis=-1)   # Batch size x 512 x 512 x 1
            focal_loss = mean_factor * tf.reduce_mean(weighted_loss)

            if self.sf_coeff > 0.0:
            	# Calculate surface loss
            	dist_map = self.onehot_distmaps(y_true)
            	y_pred_mask = tf.where(invalid_pixels, tf.zeros_like(y_pred_mask), y_pred_mask)
            	sf_loss = mean_factor * tf.reduce_mean(tf.multiply(dist_map, y_pred_mask))

            else:
                sf_loss = 0.0

            ret_loss = self.sf_coeff*sf_loss + (1-self.sf_coeff) * focal_loss

            return focal_loss

    def get_config(self):
        return {'name': self.name, 'num_classes': self.num_classes, 'gamma': self.gamma, 'beta': self.beta}

    @classmethod
    def from_config(cls, config):
        """Instantiates a `Loss` from its config (output of `get_config()`).
        Args:
            config: Output of `get_config()`.
        Returns:
            A `Loss` instance.
        """
        return cls(**config)

class AgrViGDLosses(object):
    
    def __init__(self, name='AgrVi_Dice_loss', num_classes=7):
        self.name = name
        self.num_classes = num_classes

    def __call__(self, y_true, y_pred, sample_weight=None):
        """
        https://github.com/richardaecn/class-balanced-loss
        :param y_true: batch_size x 512 x 512 x 8
        :param y_pred: batch_size x 512 x 512 x 7
        :param sample_weight:
        :return:
        """
        scope_name = 'lambda' if self.name == '<lambda>' else self.name
        graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
            y_true, y_pred, sample_weight)
        with tf.name_scope(scope_name or self.__class__.__name__), graph_ctx:
            invalid_pixels = tf.expand_dims(tf.cast(y_true[:, :, :, -1], tf.bool), -1)  # Index of invalid pixels
            n_valid_pixels = tf.size(invalid_pixels) - tf.reduce_sum(tf.cast(invalid_pixels, tf.int32))
            mean_factor = tf.cast(tf.size(invalid_pixels), tf.float32) / tf.cast(n_valid_pixels, tf.float32)

            # Calculate loss value
            # y_true_mask = tf.cast(y_true[:, :, :, :-1], tf.float32) / tf.cast(tf.reduce_sum(y_true, axis=-1, keepdims=True), tf.float32)
            y_true_mask = tf.cast(y_true[:, :, :, :-1], tf.float32)

            red_dim = np.arange(0, y_true_mask.shape.ndims - 1, 1).tolist()
            
            ref_vol = tf.reduce_sum(tf.cast(y_true[:, :, :, :-1], tf.float32), axis=[0, 1, 2])   # Out: n_classes  axis=[0, 1, 2]
            n_valid_class = tf.math.count_nonzero(ref_vol, dtype=tf.float32)
            
            # tf.print("DEBUG: ", n_valid_class == 7)

            dice_weights = tf.math.reciprocal(ref_vol)
            dice_weights = tf.where(tf.math.is_inf(dice_weights), tf.zeros_like(dice_weights), dice_weights)
            dice_weights = tf.expand_dims(dice_weights, axis=0)   # 1 x n_classes
            
            y_pred_mask = tf.nn.softmax(y_pred, axis=-1)

            intersect = tf.reduce_sum(tf.multiply(y_pred_mask, y_true_mask), axis=[1,2])   # Batch size x n_classes
            union = tf.reduce_sum(y_pred_mask) + tf.reduce_sum(y_true_mask)
            
            # Included background class => denominator > 0, numerator > 0 in any cases
            numerator = 2 * tf.reduce_sum(tf.multiply(dice_weights, intersect), axis=-1)   # Batch size
            denominator = tf.reduce_sum(tf.multiply(dice_weights, union), axis=-1)
            
            dice_sc = tf.reduce_mean(1 - numerator / denominator)
            
            return dice_sc

    def get_config(self):
        return {'name': self.name, 'num_classes': self.num_classes}

    @classmethod
    def from_config(cls, config):
        """Instantiates a `Loss` from its config (output of `get_config()`).
        Args:
            config: Output of `get_config()`.
        Returns:
            A `Loss` instance.
        """
        return cls(**config)

class AgriLrSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """docstring for AgriLrSchedule"""
    def __init__(self, initial_learning_rate, decay_steps, constant_steps=0, end_learning_rate=0.0001, power=1.0, cycle=False, name=None):
        super(AgriLrSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.constant_steps = constant_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "AgriLrSchedule") as name:
            # Step start from 1
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            end_learning_rate = tf.cast(self.end_learning_rate, dtype)
            power = tf.cast(self.power, dtype)

            step_cf = tf.cast(step, dtype)
            constant_steps_cf = tf.cast(self.constant_steps, dtype)

            global_step_recomp = tf.where(tf.less(step_cf - constant_steps_cf, 0), 0.0, step_cf - constant_steps_cf)

            decay_steps_recomp = tf.cast(self.decay_steps, dtype)
            if self.cycle:
                # Find the first multiple of decay_steps that is bigger than
                # global_step. If global_step is zero set the multiplier to 1
                multiplier = tf.cond(
                    tf.equal(global_step_recomp, 0), lambda: 1.0,
                    lambda: tf.math.ceil(global_step_recomp / self.decay_steps))
                decay_steps_recomp = tf.multiply(decay_steps_recomp, multiplier)
            else:
                # Make sure that the global_step used is not bigger than decay_steps.
                global_step_recomp = tf.minimum(global_step_recomp,
                                                      self.decay_steps)

                p = tf.divide(global_step_recomp, decay_steps_recomp)
            return tf.add(tf.multiply(initial_learning_rate - end_learning_rate,
                                tf.pow(1 - p, power)),
                end_learning_rate,
                name=name)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "constant_steps": self.constant_steps,
            "end_learning_rate": self.end_learning_rate,
            "power": self.power,
            "cycle": self.cycle,
            "name": self.name
        }
        
class VerboseFitCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        super(VerboseFitCallBack).__init__()
        self.columns = None
        self.st_time = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.st_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        current_header = list(logs.keys())
        if 'lr' in current_header:
            lr_index = current_header.index('lr')
        else:
            lr_index = len(current_header)

        if self.columns is None:
            self.columns = ['ep', 'lr'] + current_header[:lr_index] + current_header[lr_index + 1:] + ['time']
        logs_values = list(logs.values())
        current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)

        current_step = tf.cast(self.model.optimizer.iterations, tf.float32)
        current_lr = float(current_lr(current_step))

        try:
            current_step = tf.cast(self.model.optimizer.iterations, tf.float32)
            current_lr = float(current_lr(current_step))
        except:
            current_lr = float(current_lr)

        time_ep = time.time() - self.st_time
        current_values = [epoch + 1, current_lr] + logs_values[:lr_index] + logs_values[lr_index + 1:] + [time_ep]
        table = tabulate.tabulate([current_values], self.columns, tablefmt='simple', floatfmt='10.6g')
        if epoch % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

class LogBatchLoss(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LogBatchLoss).__init__()
        self.lrs_ = []
        self.losses_ = []
        self.beta = 0.98
        self.avg_loss = 0.0
        self.best_loss = 0.0

    def on_train_batch_end(self, batch, logs=None):
        current_header = list(logs.keys())

        logs_values = list(logs.values())
        current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        current_step = tf.cast(self.model.optimizer.iterations, tf.float32)

        try:
            current_lr = float(current_lr(current_step))
        except:
            current_lr = float(current_lr)
        self.lrs_.append(current_lr)

        loss_index = current_header.index('loss')
        current_loss = logs_values[loss_index]

        self.avg_loss = self.beta*self.avg_loss + (1-self.beta) * current_loss
        smoothed_loss = self.avg_loss / (1 - self.beta ** (len(self.losses_) + 1))

        if len(self.losses_) > 1 and smoothed_loss > 4*self.best_loss:
            self.model.stop_training = True
        
        if len(self.losses_) == 0 or smoothed_loss < self.best_loss:
            self.best_loss = smoothed_loss

        self.losses_.append(smoothed_loss)

        if len(self.losses_) >= args.clr_step:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if len(self.lrs_) > 0 and len(self.losses_) > 0:
            write_data = np.transpose(np.array([self.lrs_, self.losses_]))
            np.savetxt('lr_loss_log.txt', write_data)
        else:
            with open('lr_loss_log.txt', "w") as fd:
                fd.write("Run FAILED")

        print('Finished lr range test')

"""
Data processing functions
"""
def load_distance_map(file_path):
    dmap = np.load(file_path.numpy(), allow_pickle=True)['dmap'][()]
    res = np.concatenate([dmap[0], dmap[1], dmap[2], dmap[3], dmap[4], dmap[5], dmap[6]], axis=-1).astype(np.float32)
    
    return res

def augment_dev(image, target, func, sel):
    """ Augment image """
    image = tf.where(sel, func(image), image)
    target = tf.where(sel, func(target), target)

    return image, target

def parse_transform_dev(feat, target):
    # feat, target = batch
    is_flip_left_right = tf.random.uniform([], seed=1)
    is_flip_up_down = tf.random.uniform([], seed=1)
    is_rotate_90 = tf.random.uniform([], maxval=2**10, dtype=tf.int32)

    feat, target = augment_dev(feat, target, tf.image.flip_left_right, is_flip_left_right>0.5)
    feat, target = augment_dev(feat, target, tf.image.flip_up_down, is_flip_up_down>0.5)
    feat, target = augment_dev(feat, target, partial(tf.image.rot90, k=is_rotate_90), is_rotate_90 > 0)
    return feat, target

def generate_augment_dev(image, file_path):
    encode_jpeg = tf.strings.regex_full_match(file_path, tf.constant(".+jpg"))
    fn_ilr = tf.strings.regex_replace(tf.strings.regex_replace(file_path, '.jpg', '_flr.jpg'), '.png', '_flr.png')
    fn_iud = tf.strings.regex_replace(tf.strings.regex_replace(file_path, '.jpg', '_fud.jpg'), '.png', '_fud.png')
    fn_rt90 = tf.strings.regex_replace(tf.strings.regex_replace(file_path, '.jpg', '_rt90.jpg'), '.png', '_rt90.png')
    fn_rt180 = tf.strings.regex_replace(tf.strings.regex_replace(file_path, '.jpg', '_rt180.jpg'), '.png', '_rt180.png')
    fn_rt270 = tf.strings.regex_replace(tf.strings.regex_replace(file_path, '.jpg', '_rt270.jpg'), '.png', '_rt270.png')

    ilr = tf.image.flip_left_right(image)
    ilr_str = tf.where(encode_jpeg, tf.image.encode_jpeg(ilr), tf.image.encode_png(ilr))
    tf.io.write_file(fn_ilr, ilr_str)

    iud = tf.image.flip_up_down(image)
    iud_str = tf.where(encode_jpeg, tf.image.encode_jpeg(iud), tf.image.encode_png(iud))
    tf.io.write_file(fn_iud, iud_str)

    irt90 = tf.image.rot90(image, k=1)
    irt90_str = tf.where(encode_jpeg, tf.image.encode_jpeg(irt90), tf.image.encode_png(irt90))
    tf.io.write_file(fn_rt90, irt90_str)

    irt180 = tf.image.rot90(image, k=2)
    irt180_str = tf.where(encode_jpeg, tf.image.encode_jpeg(irt180), tf.image.encode_png(irt180))
    tf.io.write_file(fn_rt180, irt180_str)

    irt270 = tf.image.rot90(image, k=3)
    irt270_str = tf.where(encode_jpeg, tf.image.encode_jpeg(irt270), tf.image.encode_png(irt270))
    tf.io.write_file(fn_rt270, irt270_str)

def fn_dev_augment(image, label):
    """ Image augmentation """
    label = tf.cast(label, tf.float32)
    lr_p = tf.random.uniform((), dtype=tf.float32)
    ud_p = tf.random.uniform((), dtype=tf.float32)
    r90_v = tf.cast(tf.math.floormod(tf.random.uniform((), maxval=2**16, dtype=tf.int32), 4), tf.int32)

    image = tf.where(lr_p < 0.5, image, tf.image.flip_left_right(image))
    image = tf.where(ud_p < 0.5, image, tf.image.flip_up_down(image))
    image = tf.where(r90_v < 1, image, tf.image.rot90(image, r90_v))

    label = tf.where(lr_p < 0.5, label, tf.image.flip_left_right(label))
    label = tf.where(ud_p < 0.5, label, tf.image.flip_up_down(label))
    label = tf.where(r90_v < 1, label, tf.image.rot90(label, r90_v))

    return image, label

def parse_fn_dev(file_path):
    """
    Process file_path
    :param file_path:
    :return:
    """
    # regex_train = tf.constant('.train.')
    is_train_set = tf.strings.regex_full_match(file_path, tf.constant(".+train.+"))
    # rd_seed = tf.random.uniform([3, ])
    # tf.print(rd_seed, file_path)

    rgb_path = file_path
    nir_path = tf.strings.regex_replace(file_path, 'rgb', 'nir')
    file_name = tf.strings.regex_replace(tf.strings.split(rgb_path, sep_path)[-1], '.jpg', '.png')
    repx = sep_path + sep_path if os.name == 'nt' else sep_path
    boundaries_path = tf.strings.regex_replace(tf.strings.regex_replace(file_path, '.jpg', '.png'),
                                               "images{}rgb".format(repx), "boundaries")
    mask_path = tf.strings.regex_replace(tf.strings.regex_replace(file_path, '.jpg', '.png'),
                                         "images{}rgb".format(repx), "masks")

    # Read RGB
    rgb_image = tf.image.decode_image(tf.io.read_file(rgb_path), channels=3)
    # generate_augment_dev(rgb_image, rgb_path)

    # Read NIR
    nir_image = tf.image.decode_image(tf.io.read_file(nir_path), channels=1)
    # generate_augment_dev(nir_image, nir_path)

    # Concat rgb and nir to NRGB
    nrgb_image = tf.concat([nir_image, rgb_image], axis=2)  # 512 x 512 x 4

    # Read boundary
    bdr_image = tf.image.decode_image(tf.io.read_file(boundaries_path), channels=1)
    # generate_augment_dev(bdr_image, boundaries_path)

    # Read mask
    msk_image = tf.image.decode_image(tf.io.read_file(mask_path), channels=1)
    # generate_augment_dev(msk_image, mask_path)

    invalid_pixels = tf.logical_or(bdr_image == 0, msk_image == 0)

    # nrgb_image = tf.where(invalid_pixels, tf.zeros_like(nrgb_image), nrgb_image)
    nrgb_image = tf.image.convert_image_dtype(nrgb_image, tf.float32)
    
    # Use NDVI
    # ndvi = tf.expand_dims(tf.math.divide_no_nan(nrgb_image[:, :, 0] - nrgb_image[:, :, 1], nrgb_image[:, :, 0] + nrgb_image[:, :, 1]), axis=-1)
    # nrgb_image = tf.concat([nrgb_image, ndvi], axis=2)

    ld_labels = {0: tf.identity(invalid_pixels)}
    
    for idx in range(1, len(label_dict)):
        current_name = tf.strings.regex_replace(tf.strings.regex_replace(file_path, '.jpg', '.png'),
                                                "images{}rgb".format(repx), "labels{}{}".format(repx, label_dict[idx]))
        # Read label
        current_label = tf.image.decode_image(tf.io.read_file(current_name), channels=1)
        # generate_augment_dev(current_label, current_name)

        ld_labels[idx] = tf.logical_and(current_label > 0, tf.logical_not(invalid_pixels))
        ld_labels[0] = tf.logical_or(ld_labels[idx], ld_labels[0])
        

    ld_labels[0] = tf.logical_not(ld_labels[0])

    ret_label = tf.cast(tf.concat(
        [ld_labels[0], ld_labels[1], ld_labels[2], ld_labels[3], ld_labels[4], ld_labels[5], ld_labels[6],
        invalid_pixels], axis=2), dtype=tf.int32)  # 512 x 512 x 8

    nrgb_image.set_shape((512, 512, 4))
    ret_label.set_shape((512, 512, 8))

    nrgb_image = (nrgb_image  - mean) / std
    return nrgb_image, ret_label#, file_name

def parse_fn_test(file_path):
    """
    Process file_path
    :param file_path:
    :return:
    """
    rgb_path = file_path
    file_name = tf.strings.regex_replace(tf.strings.split(rgb_path, sep_path)[-1], '.jpg', '.png')
    nir_path = tf.strings.regex_replace(file_path, 'rgb', 'nir')

    rgb_image = tf.image.decode_image(tf.io.read_file(rgb_path), channels=3)
    nir_image = tf.image.decode_image(tf.io.read_file(nir_path), channels=1)
    nrgb_image = tf.concat([nir_image, rgb_image], axis=2)  # 512 x 512 x 4
    nrgb_image = (tf.image.convert_image_dtype(nrgb_image, tf.float32) - mean) / std

    return nrgb_image, file_name


def count_(counts, batch):
    """
    Small function for calculate mean, std
    """
    features, _ = batch
    # tmp = tf.reduce_sum(features, axis=[0, 1, 2])   # Calculate mean
    tmp = tf.reduce_sum(tf.square(features - mean), axis=[0, 1, 2])  # Calculate std
    counts += tmp
    return counts


def generate_test_label(model, loader, n_classes=7, write_folder=None):
    """ Generate label for loader with model """
    def write_image(im_arr, im_name='test'):
        if len(np.unique(im_arr)) > 3:
            print(np.unique(im_arr), end='----')
        Image.fromarray(np.uint8(im_arr), 'L').save(os.path.join(write_folder, im_name))
        #plt.imsave(os.path.join(write_folder, 'test-' + im_name), im_arr)
        load_again = np.array(Image.open(os.path.join(write_folder, im_name)))
        
        if np.sum(np.abs(im_arr - load_again)) > 0:
            print('*****  Error in here: ', im_name)
            sys.exit(0)

    if not os.path.isdir(write_folder):
        os.makedirs(write_folder, exist_ok=True)

    for feat, image_names in tqdm(loader):
        image_names = image_names.numpy()
        out_probs = model(feat)
        out_cat = np.argmax(out_probs, axis=-1)

        for idx in range(out_cat.shape[0]):
            write_image(out_cat[idx, :, :], image_names[idx].decode("utf-8") )


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='Agriculture Vision')
    parser.add_argument('--dataset', type=str, default='/mnt/SharedProject/Agriculture_Vision/dataset', required=False,
                        help='training directory (default: /mnt/SharedProject/Agriculture_Vision/dataset)')
    parser.add_argument('--dir', type=str, default='./tmp/', required=False, help='training directory (default: tmp)')
    
    parser.add_argument('--opt', type=str, default='adam', required=False,
                        help='Optimizer: Adam or SGD (default: Adam)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', required=False,
                        help='input batch size (default: 32)')
    parser.add_argument('--lr_init', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--lr_max', type=float, default=0, metavar='LR',
                        help='max learning rate in Cyclic LR(default: 0)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay (default: 5e-4)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')

    parser.add_argument('--gamma', type=float, default=0, metavar='S',
                        help='Gamma for focal loss (default: 0 - only CE)')
    parser.add_argument('--beta', type=float, default=0.9999, metavar='S',
                        help='Coefficient for Class balance focal loss(default: 0.9999)')

    parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                        help='checkpoint to resume training from (default: None)')
    
    parser.add_argument('--clr_step', type=int, default=4000, required=False, help='CLR step size (default: 4000)')
    parser.add_argument('--sf_coeff', type=float, default=0.0, required=False, help='sf_coeff (default: 0.0)')

    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--ratio_train', type=float, default=0.8, metavar='S',
                        help='Ratio of train dataset for each epoch (default: 0.8)')
    parser.add_argument('--test', type=str, default='', required=False, help='Enter testing progress (default: No)')
    parser.add_argument('--wtest', type=int, default=0, required=False, help='Generate test label or not (default: No)')
    args = parser.parse_args()
    
    is_disable = True if args.test in ['loss', 'miou', 'floyd'] else False
    experiment = Experiment(api_key="uG1BcicYOr83KvLjFEZQMrWVg",
                        project_name="agrivision", workspace="th2l-aipn",
                        auto_output_logging="simple", disabled=is_disable)
    
    print('Preparing directory {} to write results'.format(args.dir))
    os.makedirs(args.dir, exist_ok=True)
    #sys.stdout = open(args.dir + '/log_{}.txt'.format(args.test), 'w')
    with open(os.path.join(args.dir, 'command_{}.sh'.format(args.test)), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')
        f.write(experiment.get_key())
        f.write('\n')

    os.environ['PYTHONHASHSEED'] = str(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("TensorFlow version: ", tf.__version__)
    print("Use GPU: ", tf.test.is_gpu_available(), "\n")
    print('{} Loading dataset from: {}'.format(datetime.datetime.now(), args.dataset))

    if os.name == 'nt':
        root_folder = pathlib.Path('F:/Agriculture_Vision/dataset')
        sep_path = "\\"
    else:
        root_folder = pathlib.Path(args.dataset)
        sep_path = '/'

    batch_size = args.batch_size
    val_batch_size = batch_size if (batch_size < 16 and args.test in ['loss', 'miou']) else batch_size

    n_classes = 7
    train_dict = [None, None, None, None, None, None]
    train_full = []
    shuffle_size = 10000
    source_path = './data_files/'
    # source_path = './data_files/augmented/'
    for cidx in range(1, 7):
        train_dict[cidx - 1] = source_path + label_dict[cidx] + '.txt'
        ld_files = np.loadtxt(source_path + label_dict[cidx] + '.txt', dtype=str).tolist()
        shuffle_size = max(shuffle_size, len(ld_files)+3)
        # ld_lr = [x.replace('.jpg', '_flr.jpg') for x in ld_files]
        # ld_ud = [x.replace('.jpg', '_fud.jpg') for x in ld_files]
        # ld_rt90 = [x.replace('.jpg', '_rt90.jpg') for x in ld_files]
        # ld_rt180 = [x.replace('.jpg', '_rt180.jpg') for x in ld_files]
        # ld_rt270 = [x.replace('.jpg', '_rt270.jpg') for x in ld_files]

        # ld_news = ld_files + ld_lr + ld_ud + ld_rt90 + ld_rt180 + ld_rt270
        print(label_dict[cidx], len(ld_files))
        # np.savetxt(label_dict[cidx] + '.txt', ld_news, fmt="%s")

        train_full = train_full + ld_files

        # train_dict[cidx - 1] = tf.data.Dataset.from_tensor_slices(ld_files).shuffle(len(ld_files) + 5, seed=args.seed, reshuffle_each_iteration=True).repeat()
    # sys.exit(0)
    print('Number of train files in train folder: ', len(train_full))
    print("Shuffle size: ", shuffle_size)
    valid_files = sorted(glob.glob(os.path.join(root_folder, 'val/images/rgb/*.jpg')))
    test_files = sorted(glob.glob(os.path.join(root_folder, 'test/images/rgb/*.jpg')))

    # train_ds = tf.data.experimental.sample_from_datasets(train_dict, weights=None, seed=args.seed).cache()
    train_ds = tf.data.Dataset.from_tensor_slices(train_dict)
    train_full_ds = tf.data.Dataset.from_tensor_slices(train_full)

    valid_ds = tf.data.Dataset.from_tensor_slices(valid_files).cache()
    test_ds = tf.data.Dataset.from_tensor_slices(test_files).cache()

    # print("Calculating mean and std")
    # counts = tf.zeros(5, )
    # counts_update = train_full_ds.map(parse_fn_dev, num_parallel_calls=AUTOTUNE).batch(batch_size=batch_size).prefetch(AUTOTUNE).reduce(initial_state=counts, reduce_func=count_).numpy()
    # # print("Mean: ", counts_update, counts_update / (512*512*len(train_full)), len(train_full))
    # print("STD: ", counts_update, np.sqrt(counts_update / (512*512*len(train_full))), len(train_full))
    # sys.exit(0)
    # .map(fn_dev_augment, num_parallel_calls=AUTOTUNE)
    datasets = {'train': train_ds.interleave(lambda x: tf.data.TextLineDataset(x).shuffle(shuffle_size, seed=args.seed, reshuffle_each_iteration=True).map(parse_fn_dev, num_parallel_calls=AUTOTUNE).repeat(), cycle_length=6, block_length=1, num_parallel_calls=AUTOTUNE).batch(batch_size=batch_size).prefetch(AUTOTUNE),
                'train_full': train_full_ds.shuffle(len(train_full)+5, seed=args.seed, reshuffle_each_iteration=True).map(parse_fn_dev, num_parallel_calls=AUTOTUNE).batch(batch_size=batch_size).prefetch(AUTOTUNE),
                'val': valid_ds.map(parse_fn_dev, num_parallel_calls=AUTOTUNE).batch(batch_size=val_batch_size).prefetch(AUTOTUNE),
                'test': test_ds.map(parse_fn_test, num_parallel_calls=AUTOTUNE).batch(batch_size=batch_size*3).prefetch(AUTOTUNE)
    }
    num_epochs = args.epochs

    # print("Generate augment_dev")
    # for bf in tqdm(datasets['train_full']):
    #     continue
    # sys.exit(0)
    # Define model
    model = AgriVi_Segmentation(upsample_method='bilinear', output_channels=n_classes)

    if args.resume:
        if 'last_epoch_checkpoint' in args.resume:
            model = tf.keras.models.load_model(args.resume, custom_objects={'loss':loss_obj})
        else:
            # Load pre-trained weight
            model.load_weights(args.resume)
            model.summary()
            print("Number of layers: ", len(model.layers))
            model.layers[1].trainable = False

    num_trains = 12901
    n_step_train = num_trains // batch_size + 1
    STEPS_PER_EPOCH = int(n_step_train * args.ratio_train) #if args.ratio_train < 1 else None
    STEPS_PER_EPOCH = 100*(STEPS_PER_EPOCH // 100)

    if args.lr_max > 0:
        print("Use TriangularCyclicalLearningRate")
        clr_step_size = args.clr_step
        lr_schedule = tfa.optimizers.cyclical_learning_rate.TriangularCyclicalLearningRate(
                                                                initial_learning_rate=args.lr_init,
                                                                maximal_learning_rate=args.lr_max,
                                                                step_size=clr_step_size,
                                                                scale_mode="cycle") #cycle  iterations
    else:
        print("Use Polynomial Decay")
        decay_steps = 20 * STEPS_PER_EPOCH
        lr_schedule = AgriLrSchedule(initial_learning_rate=args.lr_init, decay_steps=decay_steps, constant_steps=10*STEPS_PER_EPOCH, end_learning_rate=2e-3, power=0.9, cycle=False)
        # warmup_step = 5 * STEPS_PER_EPOCH
        # poly_step = 7 * STEPS_PER_EPOCH
        # constant_step = num_epochs * STEPS_PER_EPOCH - (warmup_step + poly_step)
        # lr_schedule = AgriLrSchedule(args.lr_init, warmup_step, constant_step, poly_step)

    if args.opt == 'adam':
        optimizer = tfa.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=args.wd)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=args.momentum, decay=args.wd)

    loss_name = "CB"
    if loss_name == "CB":
	    print("Used Class-balanced cross entropy loss, sf_coeff: ", args.sf_coeff)
	    loss_obj = AgrViCBLosses(num_classes=n_classes, beta=args.beta, gamma=args.gamma, sf_coeff=args.sf_coeff)
    else:
	    print("Used old loss, not good :v")
	    loss_obj = AgrViLosses(num_classes=n_classes, type_weight="simple", alpha=0, smoothing=0, gamma=args.gamma)

    model.compile(optimizer=optimizer,
                  loss=loss_obj,
                  metrics=[AgrVimIOU(num_classes=n_classes, per_classes=False)])
    print("Number of parameters: ", model.count_params())

    # if args.resume:
    #     print("Initial evaluation on val set: ")
    #     model.evaluate(datasets['val'])

    if args.test in ['loss', 'miou']:
        print('Testing in progress')
        print('Number of valid files: ', len(valid_files))
        print('Number of test files: ', len(test_files))
        # Load model for evaluate
        print("Evaluate best: ", args.test)
        model.load_weights(os.path.join(args.dir, 'best_model_{}.h5'.format(args.test)))

        # model.evaluate(datasets['train'])
        # model.evaluate(datasets['val'])
        # sys.exit(0)
        val_mIOU = AgrVimIOU(num_classes=n_classes, per_classes=True)
        # val_cnf = AgrVimCNF(num_classes=n_classes, per_classes=True)
        for x, y in tqdm(datasets['val']):
            y_ = model(x)
            val_mIOU.update_state(y, y_)
            # val_cnf.update_state(y, y_)
        print(val_mIOU.result())
        # print(val_cnf.result())
        
        if args.wtest:
            print("Generating test labels")
            test_write_folder = os.path.join(args.dir, 'test')
            generate_test_label(model, datasets['test'], write_folder=test_write_folder)
        
    else:
        # List of callbacks
        ckpt_loss = tf.keras.callbacks.ModelCheckpoint(os.path.join(args.dir, 'best_model_loss.h5'), monitor='val_loss', verbose=0,
                                                save_best_only=True, save_weights_only=True, mode='auto')
        ckpt_miou = tf.keras.callbacks.ModelCheckpoint(os.path.join(args.dir, 'best_model_miou.h5'), monitor='val_AgrVimIOU', verbose=0,
                                                save_best_only=True, save_weights_only=True, mode='max')
        cbacks = [ckpt_loss, ckpt_miou, VerboseFitCallBack()]

        #step_count = 0
        #for feat, target in datasets['train']:
            #target_count = tf.reduce_sum(target[:, :, :, :-1], axis=[0,1,2]).numpy()
            #if np.min(target_count) < 10:
                #print(target_count / np.sum(target_count), '-----------', np.argwhere(target_count == 0))
            #else:
                #print(target_count / np.sum(target_count))
            #if step_count >= 1000:
                #print('Finished')
                #sys.exit(0)
            #else:
                #step_count += 1
        
        
        # print('Learning rate test', initializer)
        # model_history = model.fit(datasets['train'], epochs=num_epochs, steps_per_epoch=STEPS_PER_EPOCH, shuffle=True,
        #                                 validation_data=None, callbacks=[LogBatchLoss()], #
        #                                 verbose=1, validation_freq=1)
        # sys.exit(0)
        print('Training in progress')
        with experiment.train():
            try:
                model_history = model.fit(datasets['train'], epochs=num_epochs, steps_per_epoch=STEPS_PER_EPOCH, shuffle=True,
                                        validation_data=datasets['val'], callbacks=cbacks,
                                        verbose=0, validation_freq=1)
            except InterruptedExperiment as exc:
                print("Stopped Experiment")

            model.save_weights(os.path.join(args.dir, 'last_epoch_checkpoint.h5'), overwrite=True)
            model.save(os.path.join(args.dir, 'last_epoch_checkpoint_opt.h5'), include_optimizer=True, overwrite=True)
            # model = tf.keras.models.load_model(os.path.join(args.dir, 'last_epoch_checkpoint'), compile=False)

        print('Loss model: ')
        model.load_weights(os.path.join(args.dir, 'best_model_loss.h5'))
        model.evaluate(datasets['val'])
        
        print('mIOU model: ')
        model.load_weights(os.path.join(args.dir, 'best_model_miou.h5'))
        model.evaluate(datasets['val'])
        
    print('Everything ok')
    
    pass
