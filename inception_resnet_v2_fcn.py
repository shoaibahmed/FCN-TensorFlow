# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition of the Inception Resnet V2 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
slim = tf.contrib.slim

from math import ceil
import numpy as np


def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 35x35 resnet block."""
  with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
      tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
    mixed = tf.concat(3, [tower_conv, tower_conv1_1, tower_conv2_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 17x17 resnet block."""
  with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                  scope='Conv2d_0b_1x7')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                  scope='Conv2d_0c_7x1')
    mixed = tf.concat(3, [tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 8x8 resnet block."""
  with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                  scope='Conv2d_0b_1x3')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                  scope='Conv2d_0c_3x1')
    mixed = tf.concat(3, [tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def inception_resnet_v2(inputs, dropout_keep_prob, 
                        num_classes=1001, is_training=True,
                        reuse=None,
                        scope='InceptionResnetV2'):
  """Creates the Inception Resnet V2 model.
  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: float, the fraction to keep before final layer.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
  Returns:
    logits: the logits outputs of the model.
    end_points: the set of end_points from the inception model.
  """
  inputShape = inputs.get_shape().as_list()
  inputShape[0] = -1# self.batchSize # Images in batch
  inputShape[3] = num_classes

  end_points = {}

  with tf.variable_scope(scope, 'InceptionResnetV2', [inputs], reuse=reuse):
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='SAME'):

        # H x W x 32
        net = slim.conv2d(inputs, 32, 3, stride=1, padding='SAME',
                          scope='Conv2d_1a_3x3')
        end_points['Conv2d_1a_3x3'] = net
        # H x W x 32
        net = slim.conv2d(net, 32, 3, padding='SAME',
                          scope='Conv2d_2a_3x3')
        end_points['Conv2d_2a_3x3'] = net
        # H x W x 64
        net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
        end_points['Conv2d_2b_3x3'] = net
        # H/2 x W/2 x 64
        net = slim.max_pool2d(net, 3, stride=2, padding='SAME',
                              scope='MaxPool_3a_3x3')
        end_points['MaxPool_3a_3x3'] = net
        # H/2 x W/2 x 80
        net = slim.conv2d(net, 80, 1, padding='SAME',
                          scope='Conv2d_3b_1x1')
        end_points['Conv2d_3b_1x1'] = net
        # H/2 x W/2 x 192
        net = slim.conv2d(net, 192, 3, padding='SAME',
                          scope='Conv2d_4a_3x3')
        end_points['Conv2d_4a_3x3'] = net
        # H/4 x W/4 x 192
        net = slim.max_pool2d(net, 3, stride=2, padding='SAME',
                              scope='MaxPool_5a_3x3')
        end_points['MaxPool_5a_3x3'] = net

        # H/4 x W/4 x 320
        with tf.variable_scope('Mixed_5b'):
          with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
          with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                        scope='Conv2d_0b_5x5')
          with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                        scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                        scope='Conv2d_0c_3x3')
          with tf.variable_scope('Branch_3'):
            tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                         scope='AvgPool_0a_3x3')
            tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                       scope='Conv2d_0b_1x1')
          net = tf.concat(3, [tower_conv, tower_conv1_1,
                              tower_conv2_2, tower_pool_1])

        end_points['Mixed_5b'] = net
        net = slim.repeat(net, 10, block35, scale=0.17)

        # H/8 x W/8 x 1024
        with tf.variable_scope('Mixed_6a'):
          with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 384, 3, stride=2, padding='SAME',
                                     scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                        scope='Conv2d_0b_3x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                        stride=2, padding='SAME',
                                        scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_2'):
            tower_pool = slim.max_pool2d(net, 3, stride=2, padding='SAME',
                                         scope='MaxPool_1a_3x3')
          net = tf.concat(3, [tower_conv, tower_conv1_2, tower_pool])

        end_points['Mixed_6a'] = net
        net = slim.repeat(net, 20, block17, scale=0.10)

        # # Auxillary tower
        # with tf.variable_scope('AuxLogits'):
        #   aux = slim.avg_pool2d(net, 5, stride=3, padding='SAME',
        #                         scope='Conv2d_1a_3x3')
        #   aux = slim.conv2d(aux, 128, 1, scope='Conv2d_1b_1x1')
        #   aux = slim.conv2d(aux, 768, aux.get_shape()[1:3],
        #                     padding='SAME', scope='Conv2d_2a_5x5')
        #   aux = slim.flatten(aux)
        #   aux = slim.fully_connected(aux, num_classes, activation_fn=None,
        #                              scope='Logits')
        #   end_points['AuxLogits'] = aux

        # H/16 x W/16 x 2016
        with tf.variable_scope('Mixed_7a'):
          with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                       padding='SAME', scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_1'):
            tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                        padding='SAME', scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_2'):
            tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                        scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                        padding='SAME', scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_3'):
            tower_pool = slim.max_pool2d(net, 3, stride=2, padding='SAME',
                                         scope='MaxPool_1a_3x3')
          net = tf.concat(3, [tower_conv_1, tower_conv1_1,
                              tower_conv2_2, tower_pool])

        end_points['Mixed_7a'] = net

        net = slim.repeat(net, 9, block8, scale=0.20)
        net = block8(net, activation_fn=None)

        # Dropout
        net = slim.dropout(net, dropout_keep_prob, scope='Dropout_1')

        # H/16 x W/16 x 1536
        net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
        end_points['Conv2d_7b_1x1'] = net

        # Dropout
        net = slim.dropout(net, dropout_keep_prob, scope='Dropout_2')

        # with tf.variable_scope('Logits'):
        #   end_points['PrePool'] = net
        #   net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='SAME',
        #                         scope='AvgPool_1a_8x8')
        #   net = slim.flatten(net)

        #   net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
        #                      scope='Dropout')

        #   end_points['PreLogitsFlatten'] = net
        #   logits = slim.fully_connected(net, num_classes, activation_fn=None,
        #                                 scope='Logits')
        #   end_points['Logits'] = logits
        #   end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')  

  # Create classification layer
  regularizer = slim.l2_regularizer(0.0005)
  score_fr = slim.conv2d(net, num_classes, 1, activation_fn=None, weights_regularizer=regularizer,
                          weights_initializer=tf.truncated_normal_initializer(stddev=(2 / 1536)**0.5), scope='score_fr')
  # score_fr = _score_layer(net, "score_fr", num_classes)

  # Upscaling
  # pred_upconv = slim.conv2d_transpose(net, num_classes,
  #                                      kernel_size = [3, 3],
  #                                      stride = 16,
  #                                      padding='SAME')
  # pred_upconv = _upscore_layer(net,
  #                               shape=tf.shape(inputs),
  #                               num_classes=num_classes,
  #                               name='predUpConv',
  #                               ksize=32, stride=16)

  # 16 -> 8
  upscore2 = _upscore_layer(score_fr,
                            shape=tf.shape(end_points['Mixed_6a']),
                            num_classes=num_classes,
                            name='upscore2',
                            ksize=4, stride=2)
  score_pool4 = slim.conv2d(end_points['Mixed_6a'], num_classes, 1, activation_fn=None, weights_regularizer=regularizer,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01), scope='score_pool4')
  # score_pool4 = _score_layer(end_points['Mixed_6a'], "score_pool4", num_classes)
  fuse_pool4 = tf.add(upscore2, score_pool4)

  # 8->4
  upscore4 = _upscore_layer(fuse_pool4,
                            shape=tf.shape(end_points['MaxPool_5a_3x3']),
                            num_classes=num_classes,
                            name='upscore4',
                            ksize=4, stride=2)
  score_pool3 = slim.conv2d(end_points['MaxPool_5a_3x3'], num_classes, 1, activation_fn=None, weights_regularizer=regularizer,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.001), scope='score_pool3')
  # score_pool3 = _score_layer(end_points['MaxPool_5a_3x3'], "score_pool3", num_classes)
  fuse_pool3 = tf.add(upscore4, score_pool3)

  # 4->2
  upscore8 = _upscore_layer(fuse_pool3,
                            shape=tf.shape(end_points['MaxPool_3a_3x3']),
                            num_classes=num_classes,
                            name='upscore8',
                            ksize=4, stride=2)
  score_pool2 = slim.conv2d(end_points['MaxPool_3a_3x3'], num_classes, 1, activation_fn=None, weights_regularizer=regularizer,
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.0001), scope='score_pool2')
  # score_pool2 = _score_layer(end_points['MaxPool_3a_3x3'], "score_pool2", num_classes)
  fuse_pool2 = tf.add(upscore8, score_pool2)

  # 2->1
  upscore16 = _upscore_layer(fuse_pool2,
                            shape=tf.shape(end_points['Conv2d_1a_3x3']),
                            num_classes=num_classes,
                            name='upscore16',
                            ksize=4, stride=2)

  # logits = tf.reshape(upscore16, (-1, num_classes))
  # epsilon = tf.constant(value=1e-4)
  # softmax = tf.nn.softmax(logits + epsilon)
  # probabilities = tf.reshape(softmax, inputShape, name='probabilities')
  probabilities = tf.nn.softmax(upscore16, name='probabilities')

  return probabilities, upscore16, end_points

inception_resnet_v2.default_image_size = 299


def inception_resnet_v2_arg_scope(weight_decay=0.00004,
                                  batch_norm_decay=0.9997,
                                  batch_norm_epsilon=0.001):
  """Yields the scope with the default parameters for inception_resnet_v2.
  Args:
    weight_decay: the weight decay for weights variables.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.
  Returns:
    a arg_scope with the parameters needed for inception_resnet_v2.
  """
  # Set weight_decay for weights in conv2d and fully_connected layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params) as scope:
      return scope


### From FCN-VGG
def get_deconv_filter(f_shape):
  width = f_shape[0]
  height = f_shape[1]
  f = ceil(width/2.0)
  c = (2 * f - 1 - f % 2) / (2.0 * f)
  bilinear = np.zeros([f_shape[0], f_shape[1]])
  for x in range(width):
      for y in range(height):
          value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
          bilinear[x, y] = value
  weights = np.zeros(f_shape)
  for i in range(f_shape[2]):
      weights[:, :, i, i] = bilinear

  init = tf.constant_initializer(value=weights,
                                 dtype=tf.float32)
  return tf.get_variable(name="up_filter", initializer=init,
                         shape=weights.shape)


def _upscore_layer(bottom, shape,
                   num_classes, name,
                   ksize=4, stride=2):
  strides = [1, stride, stride, 1]
  with tf.variable_scope(name):
    in_features = bottom.get_shape()[3].value

    if shape is None:
      # Compute shape out of Bottom
      in_shape = tf.shape(bottom)

      h = ((in_shape[1] - 1) * stride) + 1
      w = ((in_shape[2] - 1) * stride) + 1
      new_shape = [in_shape[0], h, w, num_classes]
    else:
      new_shape = [shape[0], shape[1], shape[2], num_classes]
    output_shape = tf.pack(new_shape)

    f_shape = [ksize, ksize, num_classes, in_features]

    # create
    num_input = ksize * ksize * in_features / stride
    stddev = (2 / num_input)**0.5

    weights = get_deconv_filter(f_shape)
    deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                    strides=strides, padding='SAME')
  return deconv

# def _score_layer(bottom, name, num_classes):
#   with tf.variable_scope(name) as scope:
#     # get number of input channels
#     in_features = bottom.get_shape()[3].value
#     shape = [1, 1, in_features, num_classes]
#     # He initialization Sheme
#     if name == "score_fr":
#         num_input = in_features
#         stddev = (2 / num_input)**0.5
#     elif name == "score_pool4":
#         stddev = 0.001
#     elif name == "score_pool3":
#         stddev = 0.0001
#     elif name == "score_pool2":
#         stddev = 0.00001
#     elif name == "score_pool1":
#         stddev = 0.000001
#     # Apply convolution
#     w_decay = 5e-4
#     weights = _variable_with_weight_decay(shape, stddev, w_decay)
#     conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
#     # Apply bias
#     conv_biases = _bias_variable([num_classes], constant=0.0)
#     bias = tf.nn.bias_add(conv, conv_biases)

#     return bias

# def _variable_with_weight_decay(shape, stddev, wd):
#     """Helper to create an initialized Variable with weight decay.

#     Note that the Variable is initialized with a truncated normal
#     distribution.
#     A weight decay is added only if one is specified.

#     Args:
#       name: name of the variable
#       shape: list of ints
#       stddev: standard deviation of a truncated Gaussian
#       wd: add L2Loss weight decay multiplied by this float. If None, weight
#           decay is not added for this Variable.

#     Returns:
#       Variable Tensor
#     """
    
#     initializer = tf.truncated_normal_initializer(stddev=stddev)
#     var = tf.get_variable('weights', shape=shape,
#                           initializer=initializer)
#     # var = tf.get_variable(name="weights", shape=shape, 
#     #                       initializer=tf.contrib.layers.xavier_initializer())

#     if wd and (not tf.get_variable_scope().reuse):
#         weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
#         tf.add_to_collection('losses', weight_decay)
#     return var

# def _bias_variable(shape, constant=0.0):
#     initializer = tf.constant_initializer(constant)
#     return tf.get_variable(name='biases', shape=shape,
#                            initializer=initializer)