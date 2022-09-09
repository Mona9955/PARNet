from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
slim = tf.contrib.slim


# pylint: disable=g-long-lambda
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
# inception-v2 network from https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v2.py

def inception_arg_scope(
    weight_decay=0.00004,
    use_batch_norm=True,
    batch_norm_decay=0.9997,
    batch_norm_epsilon=0.001,
    activation_fn=tf.nn.relu,
    batch_norm_updates_collections=tf.compat.v1.GraphKeys.UPDATE_OPS,
    batch_norm_scale=False):
  """Defines the default arg scope for inception models.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
    use_batch_norm: "If `True`, batch_norm is applied after each convolution.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
    activation_fn: Activation function for conv2d.
    batch_norm_updates_collections: Collection for the update ops for
      batch norm.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
  Returns:
    An `arg_scope` to use for the inception models.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': batch_norm_updates_collections,
      # use fused batch norm if possible.
      'fused': None,
      'scale': batch_norm_scale,
  }
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer = slim.variance_scaling_initializer(),
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc

def inception_v2_base(inputs,
                      final_endpoint='Mixed_5c',
                      min_depth=16,
                      depth_multiplier=1.0,
                      use_separable_conv=True,
                      data_format='NHWC',
                      include_root_block=True,
                      scope=None,
                      reuse=None):
  """Inception v2 (6a2).
  Constructs an Inception v2 network from inputs to the given final endpoint.
  This method can construct the network up to the layer inception(5b) as
  described in http://arxiv.org/abs/1502.03167.
  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c', 'Mixed_4a',
      'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_5a', 'Mixed_5b',
      'Mixed_5c']. If include_root_block is False, ['Conv2d_1a_7x7',
      'MaxPool_2a_3x3', 'Conv2d_2b_1x1', 'Conv2d_2c_3x3', 'MaxPool_3a_3x3'] will
      not be available.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    use_separable_conv: Use a separable convolution for the first layer
      Conv2d_1a_7x7. If this is False, use a normal convolution instead.
    data_format: Data format of the activations ('NHWC' or 'NCHW').
    include_root_block: If True, include the convolution and max-pooling layers
      before the inception modules. If False, excludes those layers.
    scope: Optional variable_scope.
  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.
  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0
  """

  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}

  # Used to find thinned depths for each layer.
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  if data_format != 'NHWC' and data_format != 'NCHW':
    raise ValueError('data_format must be either NHWC or NCHW.')
  if data_format == 'NCHW' and use_separable_conv:
    raise ValueError(
        'separable convolution only supports NHWC layout. NCHW data format can'
        ' only be used when use_separable_conv is False.'
    )

  concat_dim = 3 if data_format == 'NHWC' else 1
  with tf.compat.v1.variable_scope(scope, 'InceptionV2', [inputs], reuse=reuse):
    with slim.arg_scope(
        [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
        stride=1,
        padding='SAME',
        data_format=data_format):

      net = inputs
      if include_root_block:
        # Note that sizes in the comments below assume an input spatial size of
        # 224x224, however, the inputs can be of any size greater 32x32.

        # 224 x 224 x 3
        end_point = 'Conv2d_1a_7x7'

        if use_separable_conv:
          # depthwise_multiplier here is different from depth_multiplier.
          # depthwise_multiplier determines the output channels of the initial
          # depthwise conv (see docs for tf.nn.separable_conv2d), while
          # depth_multiplier controls the # channels of the subsequent 1x1
          # convolution. Must have
          #   in_channels * depthwise_multipler <= out_channels
          # so that the separable convolution is not overparameterized.
          depthwise_multiplier = min(int(depth(64) / 3), 8)
          net = slim.separable_conv2d(
              inputs,
              depth(64), [7, 7],
              depth_multiplier=depthwise_multiplier,
              stride=2,
              padding='SAME',
              weights_initializer=trunc_normal(1.0),
              scope=end_point)
        else:
          # Use a normal convolution instead of a separable convolution.
          net = slim.conv2d(
              inputs,
              depth(64), [7, 7],
              stride=2,
              weights_initializer=trunc_normal(1.0),
              scope=end_point)
        end_points[end_point] = net
        # if end_point == final_endpoint:
        #   return net, end_points
        # 112 x 112 x 64
        end_point = 'MaxPool_2a_3x3'
        net = slim.max_pool2d(net, [3, 3], scope=end_point, stride=2)
        end_points[end_point] = net
        # if end_point == final_endpoint:
        #   return net, end_points
        # 56 x 56 x 64
        end_point = 'Conv2d_2b_1x1'
        net = slim.conv2d(
            net,
            depth(64), [1, 1],
            scope=end_point,
            weights_initializer=trunc_normal(0.1))
        end_points[end_point] = net
        # if end_point == final_endpoint:
        #   return net, end_points
        # 56 x 56 x 64
        end_point = 'Conv2d_2c_3x3'
        net = slim.conv2d(net, depth(192), [3, 3], scope=end_point)
        end_points[end_point] = net
        # if end_point == final_endpoint:
        #   return net, end_points
        # 56 x 56 x 192
        end_point = 'MaxPool_3a_3x3'
        net = slim.max_pool2d(net, [3, 3], scope=end_point, stride=2)
        end_points[end_point] = net
        # if end_point == final_endpoint:
        #   return net, end_points

      # 28 x 28 x 192
      # Inception module.
      end_point = 'Mixed_3b'
      with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.compat.v1.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(64), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.compat.v1.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.compat.v1.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(32), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        # if end_point == final_endpoint: return net, end_points
      # 28 x 28 x 256
      end_point = 'Mixed_3c'
      with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.compat.v1.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.compat.v1.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.compat.v1.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        # if end_point == final_endpoint: return net, end_points
      # 28 x 28 x 320
      end_point = 'Mixed_4a'
      with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, depth(160), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3')
        with tf.compat.v1.variable_scope('Branch_1'):
          branch_1_mixed_4a = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1_mixed_4a = slim.conv2d(
              branch_1_mixed_4a, depth(96), [3, 3], scope='Conv2d_0b_3x3')

          branch_1 = slim.conv2d(
              branch_1_mixed_4a, depth(96), [3, 3], stride=2, scope='Conv2d_1a_3x3')
        with tf.compat.v1.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(
              net, [3, 3], stride=2, scope='MaxPool_1a_3x3')
        net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2])
        end_points[end_point] = net
        # if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4b'
      with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(224), [1, 1], scope='Conv2d_0a_1x1')
        with tf.compat.v1.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(64), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(
              branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
        with tf.compat.v1.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.compat.v1.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        # if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4c'
      with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.compat.v1.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.compat.v1.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(128), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.compat.v1.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        # if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4d'
      with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
        with tf.compat.v1.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(160), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.compat.v1.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(160), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(160), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.compat.v1.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        # if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_4e'
      with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(96), [1, 1], scope='Conv2d_0a_1x1')
        with tf.compat.v1.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(192), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.compat.v1.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(160), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(192), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.compat.v1.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(96), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        # if end_point == final_endpoint: return net, end_points
      # 14 x 14 x 576
      end_point = 'Mixed_5a'
      with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(
              net, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, depth(192), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3')
        with tf.compat.v1.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1_mixed_5a = slim.conv2d(branch_1, depth(256), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_1 = slim.conv2d(branch_1_mixed_5a, depth(256), [3, 3], stride=2,
                                 scope='Conv2d_1a_3x3')
        with tf.compat.v1.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2,
                                     scope='MaxPool_1a_3x3')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2])
        end_points[end_point] = net
        # if end_point == final_endpoint: return net, end_points
      # 7 x 7 x 1024
      end_point = 'Mixed_5b'
      with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
        with tf.compat.v1.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.compat.v1.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(160), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.compat.v1.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        # if end_point == final_endpoint: return net, end_points
      # 7 x 7 x 1024
      end_point = 'Mixed_5c'
      with tf.compat.v1.variable_scope(end_point):
        with tf.compat.v1.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
        with tf.compat.v1.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                 scope='Conv2d_0b_3x3')
        with tf.compat.v1.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(
              net, depth(192), [1, 1],
              weights_initializer=trunc_normal(0.09),
              scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.compat.v1.variable_scope('Branch_3'):
          branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(128), [1, 1],
              weights_initializer=trunc_normal(0.1),
              scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        # if end_point == final_endpoint: return net, end_points
        return branch_1_mixed_5a, end_points['Mixed_5c']
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v2(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 min_depth=16,
                 depth_multiplier=1.0,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV2',
                 global_pool=False):
  """Inception v2 model for classification.
  Constructs an Inception v2 network for classification as described in
  http://arxiv.org/abs/1502.03167.
  The default image size used to train this network is 224x224.
  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    is_training: whether is training or not.
    dropout_keep_prob: the percentage of activation values that are retained.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is of
        shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    global_pool: Optional boolean flag to control the avgpooling before the
      logits layer. If false or unset, pooling is done with a fixed window
      that reduces default-sized inputs to 1x1, while larger inputs lead to
      larger outputs. If true, any input size is pooled down to 1x1.
  Returns:
    net: a Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped-out input to the logits layer
      if num_classes is 0 or None.
    end_points: a dictionary from components of the network to the corresponding
      activation.
  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0
  """
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  # Final pooling and prediction
  with tf.compat.v1.variable_scope(
      scope, 'InceptionV2', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_v2_base(
          inputs, scope=scope, min_depth=min_depth,
          depth_multiplier=depth_multiplier)
      with tf.compat.v1.variable_scope('Logits'):
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(
              input_tensor=net, axis=[1, 2], keepdims=True, name='global_pool')
          end_points['global_pool'] = net
        else:
          # Pooling with a fixed kernel size.
          kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
          net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                scope='AvgPool_1a_{}x{}'.format(*kernel_size))
          end_points['AvgPool_1a'] = net
        if not num_classes:
          return net, end_points
        # 1 x 1 x 1024
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        end_points['PreLogits'] = net
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points
inception_v2.default_image_size = 224


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """Define kernel size which is automatically reduced for small input.
  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are is large enough.
  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]
  Returns:
    a tensor with the kernel size.
  TODO(jrru): Make this function work with unknown shapes. Theoretically, this
  can be done with the code below. Problems are two-fold: (1) If the shape was
  known, it will be lost. (2) inception.slim.ops._two_element_tuple cannot
  handle tensors that define the kernel size.
      shape = tf.shape(input_tensor)
      return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                         tf.minimum(shape[2], kernel_size[1])])
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out


inception_v2_arg_scope = inception_arg_scope

if __name__=='__main__':
    import numpy as np
    inputs = tf.compat.v1.placeholder(tf.float32, [2, 224, 224, 3])
    with slim.arg_scope(inception_arg_scope()):
        branch_1_mixed_4a, end_points = inception_v2_base(inputs, scope='InceptionV2')
    var_restore = tf.compat.v1.trainable_variables(scope='InceptionV2')
    loader = tf.compat.v1.train.Saver(var_list=var_restore)
    with tf.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        loader.restore(sess, './inception_v2.ckpt')

        z = sess.run(branch_1_mixed_4a, feed_dict={inputs:np.random.normal(size=[2, 224, 224, 3])})
        print(z.shape)
