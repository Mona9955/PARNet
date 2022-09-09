import tensorflow as tf
from inception_v2 import *
# from skel_velocity_model import Part_Action_Rec
# from skel_velocity_model_v import Part_Action_Rec
from skel_velocity_model import Part_Action_Rec
import config_2d as cfg
slim = tf.contrib.slim
from tensorflow.contrib.layers import xavier_initializer
frames_seq_num = 16

def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i, dim in enumerate(static):
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret



# input: bs, t, h, w, c
# output: bs, t, h*w, c//2
def transpose_cnn(input, reuse, regularizer, scope='trans_cnn'):
     with tf.compat.v1.variable_scope(scope, 'trans_cnn', [input], reuse=reuse):
         bs, t, h, w, c = shape_list(input)
         map_0 = tf.reshape(input, [bs, t, h*w, c])
         map_0 = slim.conv2d(map_0, c//2, [1,1], normalizer_fn=None)
         map_0 = slim.conv2d(map_0, c//2, [3,1], normalizer_fn=None, activation_fn=None)

         return map_0


# input: bs, t, h, w, c
# output: bs, c, h*w, t//2
def transpose_global(input, reuse, scope='trans_cnn'):
     with tf.compat.v1.variable_scope(scope, 'trans_cnn', [input], reuse=reuse):
         bs, t, h, w, c = shape_list(input)
         trans = slim.conv2d(tf.reshape(input, [bs, t, h*w, c]), c // 2, [1, 1], normalizer_fn=None)
         trans = tf.transpose(trans, [0, 3, 2, 1])  #bs, c, h*w, t
         trans = slim.conv2d(trans, t // 2, [3, 1], normalizer_fn=None, activation_fn=None)
         return trans

# inputs: bs, 16, 224, 224, 3
def Act_2D_Model(inputs, coords, num_classes, name='InceptionV2', is_training=True, reuse=None, dropout_keep_prob=0.5, cell_dropout_rate=0.2):

    with slim.arg_scope(inception_arg_scope(weight_decay=cfg.REGULARIZATION_RATE)):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            regularizer = slim.l2_regularizer(cfg.REGULARIZATION_RATE)
            bs, t, h, w, c = shape_list(inputs)
            inputs = tf.reshape(inputs, [bs*t, h, w, c])

            branch_1_mixed_5a, end_points = inception_v2_base(inputs, scope='InceptionV2', reuse=reuse)  # bs*16, 14, 14, 256    bs*16, 7, 7, 1024

            body_out, velocity_out, fu_output, fusion_attention_array, person_att_array= Part_Action_Rec(coords, branch_1_mixed_5a, is_training=is_training, cell_dropout_rate=cell_dropout_rate, reuse=reuse)

            end_points = tf.reshape(end_points, [bs, frames_seq_num, 7, 7, 1024])
            # end_points = motion_cnn(end_points, regularizer, reuse=reuse)
            # end_points = tf.reshape(end_points, [bs * frames_seq_num, 7, 7, 512])
            # inception_ave = tf.nn.avg_pool2d(end_points, [1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME')

            end_points = transpose_cnn(end_points, reuse=reuse, regularizer=regularizer)   #bs, t, 49, 512
            # end_points = transpose_global(end_points, reuse=reuse)  #bs, 512, 49, t//2
            # end_points = pure_conv(end_points, out_channel=1, reuse=reuse)
            inception_ave = tf.reduce_mean(end_points, axis=2)
            inception_ave = tf.nn.dropout(inception_ave, rate=1-dropout_keep_prob)
            inception_ave = tf.reduce_mean(inception_ave, axis=1)

            output_all = tf.concat([inception_ave, body_out, velocity_out, fu_output], axis=1)
            # output_all = slim.fully_connected(output_all, 1024, activation_fn=tf.nn.relu, reuse=reuse, weights_regularizer=regularizer, scope='output_all_class_0')
            output_all = slim.fully_connected(output_all, num_classes, activation_fn=None, reuse=reuse, weights_regularizer=regularizer, scope='output_all_class')

            output_cnn = slim.fully_connected(inception_ave, num_classes, activation_fn=None, reuse=reuse, weights_regularizer=regularizer, scope='output_cnn_class')

            output_body = slim.fully_connected(body_out, num_classes, activation_fn=None, reuse=reuse, weights_regularizer=regularizer, scope='output_body_class')

            output_velocity = slim.fully_connected(velocity_out, num_classes, activation_fn=None, reuse=reuse, weights_regularizer=regularizer, scope='output_velocity_class')

            output_fu = slim.fully_connected(fu_output, num_classes, activation_fn=None, reuse=reuse, weights_regularizer=regularizer, scope='output_fu_class')
            # return output_all, output_cnn, output_body, output_velocity, output_fu, fusion_attention_array, person_attention_array
            return output_all, output_cnn, output_body, output_velocity, output_fu, fusion_attention_array
