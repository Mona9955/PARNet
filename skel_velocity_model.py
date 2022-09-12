import tensorflow as tf
import numpy as np
import os
import config_2d as cfg
from inception_v2 import *
from tensorflow.contrib import slim
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.ops import rnn_cell_impl
import collections
from tensorflow.python.keras.utils import tf_utils
LayerRNNCell = rnn_cell_impl.LayerRNNCell  # pylint: disable=invalid-name

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in cfg.GPUS])

max_people = cfg.MAX_PEOPLE
max_frames = 16
max_points = cfg.MAX_POINTS


left_arm = [3,4,5]
right_arm = [0,1,2]
left_leg = [9,10,11]
right_leg = [6,7,8]
body = [12,13,0,3]

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

LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))

class LstmCellDropout(LayerRNNCell):
    def __init__(self, num_units, regularizer=None, forget_bias=1.0, cell_dropout=0.2, scope=None, layernorm=True):
        super(LstmCellDropout, self).__init__()
        self._num_units = num_units
        self._forget_bias = forget_bias
        # self.cell_dropout = cell_dropout
        self.regularizer = regularizer
        self.scope = scope
        self.use_layernorm = layernorm

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units))

    @property
    def output_size(self):
        return self._num_units

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                             str(inputs_shape))
        input_depth = inputs_shape[-1]
        h_depth = self._num_units
        with tf.compat.v1.variable_scope(self.scope):
            self._kernel = tf.compat.v1.get_variable(
                'kernel', shape=[input_depth + h_depth, 4 * self._num_units], initializer=tf.initializers.orthogonal(), regularizer=self.regularizer)
            self._bias = tf.compat.v1.get_variable(
                'bias', shape=[4 * self._num_units], initializer=tf.zeros_initializer())
            self.built = True

    def call(self, inputs, state):
        c, h = state
        gate_inputs = tf.matmul(tf.concat([inputs, h], axis=-1), self._kernel)
        if self.use_layernorm:
            gate_inputs = tf.contrib.layers.layer_norm(gate_inputs)
        gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)
        i, f, o, g = tf.split(gate_inputs, num_or_size_splits=4, axis=-1)
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)  # (f + forget_bias)
        o = tf.sigmoid(o)
        g = tf.nn.tanh(g)
        # new_c = tf.add(tf.multiply(f, c), tf.nn.dropout(tf.multiply(i, g), rate=self.cell_dropout))
        new_c = tf.add(tf.multiply(f, c), tf.multiply(i, g))
        new_h = tf.multiply(o, tf.nn.tanh(c))

        # new_h = tf.nn.dropout(new_h, rate=0.2) #1-self.dropout_keep_prob
        # new_c = tf.nn.dropout(new_c, rate=0.2)
        new_state = LSTMStateTuple(new_c, new_h)
        return new_h, new_state


def Part_Action_Rec(coords, cnn_map, is_training=True, decoder_size=cfg.DECODER_SIZE, lstm_size=cfg.LSTM_SIZE, attn_len=5, part_hidden_size = 32, part_decoder_size = 100, cell_dropout_rate=0.2, reuse=tf.compat.v1.AUTO_REUSE):

    with tf.compat.v1.variable_scope('att_rnn_act_rec', reuse=reuse) as scope:
        regularizer = slim.l2_regularizer(cfg.REGULARIZATION_RATE)
        with slim.arg_scope(inception_arg_scope(weight_decay=cfg.REGULARIZATION_RATE, use_batch_norm=False)):
            with slim.arg_scope([slim.dropout, slim.batch_norm], is_training=is_training):

                bs, frame, people, joints, dim = shape_list(coords)
                cnn_feature = tf.reshape(cnn_map, [bs, frame, 14, 14, 256])  # bs, 16, 7, 7, 512
                cnn_h, cnn_w, cnn_c = shape_list(cnn_feature)[-3:]
                left_arm_flatter = tf.reshape(tf.gather(coords, left_arm, axis=3), [bs, frame, people, 6])
                right_arm_flatter = tf.reshape(tf.gather(coords, right_arm, axis=3), [bs, frame, people, 6])
                left_leg_flatter = tf.reshape(tf.gather(coords, left_leg, axis=3), [bs, frame, people, 6])
                right_leg_flatter = tf.reshape(tf.gather(coords, right_leg, axis=3), [bs, frame, people, 6])
                body_flatter = tf.reshape(tf.gather(coords, body, axis=3), [bs, frame, people, 8])


                left_arm_enco = slim.fully_connected(left_arm_flatter, part_hidden_size, activation_fn=tf.nn.relu, biases_initializer=None, reuse=tf.compat.v1.AUTO_REUSE, scope='joints_enco_0')
                left_arm_enco = slim.fully_connected(left_arm_enco, part_decoder_size, activation_fn=tf.nn.tanh, reuse=tf.compat.v1.AUTO_REUSE, scope='joints_enco_1')
                right_arm_enco = slim.fully_connected(right_arm_flatter, part_hidden_size, activation_fn=tf.nn.relu, biases_initializer=None, reuse=True, scope='joints_enco_0')
                right_arm_enco = slim.fully_connected(right_arm_enco, part_decoder_size, activation_fn=tf.nn.tanh, reuse=True, scope='joints_enco_1')
                left_leg_enco = slim.fully_connected(left_leg_flatter, part_hidden_size, activation_fn=tf.nn.relu, biases_initializer=None, reuse=True, scope='joints_enco_0')
                left_leg_enco = slim.fully_connected(left_leg_enco, part_decoder_size, activation_fn=tf.nn.tanh, reuse=True, scope='joints_enco_1')
                right_leg_enco = slim.fully_connected(right_leg_flatter, part_hidden_size, activation_fn=tf.nn.relu, biases_initializer=None, reuse=True, scope='joints_enco_0')
                right_leg_enco = slim.fully_connected(right_leg_enco, part_decoder_size, activation_fn=tf.nn.tanh, reuse=True, scope='joints_enco_1')
                body_enco = slim.fully_connected(body_flatter, part_hidden_size, activation_fn=tf.nn.relu, biases_initializer=None, scope='body_0')
                body_enco = slim.fully_connected(body_enco, part_decoder_size, activation_fn=tf.nn.tanh, scope='body_1')

                enco_coords = tf.concat([left_arm_enco, right_arm_enco, left_leg_enco, right_leg_enco, body_enco], axis=-1)  #bs, frame, people, part_decoder_size*5
                with tf.compat.v1.variable_scope('enco_fuse'):
                    W_0 = tf.compat.v1.get_variable('W_fuse', [5*part_decoder_size, decoder_size], initializer=xavier_initializer(), regularizer=regularizer)
                fused_enco_coords = tf.matmul(enco_coords, W_0)    #bs, frame, people, 512

                def rnn_cell_wrapper(scope=None):
                    rnn_cell = LstmCellDropout(lstm_size, regularizer=regularizer, cell_dropout=cell_dropout_rate, scope=scope)
                    cell_wrapper = tf.contrib.rnn.AttentionCellWrapper(rnn_cell, attn_length=attn_len, state_is_tuple=True)
                    return cell_wrapper

                body_fw_cell = rnn_cell_wrapper(scope='body_fw')
                body_bw_cell = rnn_cell_wrapper(scope='body_bw')
                fusion_cell = rnn_cell_wrapper(scope='cnn_skel_fusion')
                velocity_fw_cell = rnn_cell_wrapper(scope='velocity_fw')
                velocity_bw_cell = rnn_cell_wrapper(scope='velocity_bw')


                with tf.compat.v1.variable_scope('att_lstm'):
                    with tf.compat.v1.variable_scope("init_mean"):
                        W_init_c_fw = tf.compat.v1.get_variable('W_init_c_fw', [decoder_size, lstm_size], initializer=xavier_initializer(), regularizer=regularizer)
                        W_init_h_fw = tf.compat.v1.get_variable('W_init_h_fw', [decoder_size, lstm_size], initializer=xavier_initializer(), regularizer=regularizer)
                        W_init_c_bw = tf.compat.v1.get_variable('W_init_c_bw', [decoder_size, lstm_size], initializer=xavier_initializer(), regularizer=regularizer)
                        W_init_h_bw = tf.compat.v1.get_variable('W_init_h_bw', [decoder_size, lstm_size], initializer=xavier_initializer(), regularizer=regularizer)
                        W_v_init_c_fw = tf.compat.v1.get_variable('W_v_init_c_fw', [decoder_size, lstm_size], initializer=xavier_initializer(), regularizer=regularizer)
                        W_v_init_h_fw = tf.compat.v1.get_variable('W_v_init_h_fw', [decoder_size, lstm_size], initializer=xavier_initializer(), regularizer=regularizer)
                        W_v_init_c_bw = tf.compat.v1.get_variable('W_v_init_c_bw', [decoder_size, lstm_size], initializer=xavier_initializer(), regularizer=regularizer)
                        W_v_init_h_bw = tf.compat.v1.get_variable('W_v_init_h_bw', [decoder_size, lstm_size], initializer=xavier_initializer(), regularizer=regularizer)
                        W_init_c_fu = tf.compat.v1.get_variable('W_init_c_fu', [cnn_c, lstm_size], initializer=xavier_initializer(), regularizer=regularizer)
                        W_init_h_fu = tf.compat.v1.get_variable('W_init_h_fu', [cnn_c, lstm_size], initializer=xavier_initializer(), regularizer=regularizer)

                    with tf.compat.v1.variable_scope("attention_x"):
                        W = tf.compat.v1.get_variable('W', [decoder_size, decoder_size], initializer=xavier_initializer(), regularizer=regularizer)
                    with tf.compat.v1.variable_scope("attention_h"):
                        W_h = tf.compat.v1.get_variable('W_h', [lstm_size, decoder_size], initializer=xavier_initializer(), regularizer=regularizer)
                    with tf.compat.v1.variable_scope("att"):
                        W_att = tf.compat.v1.get_variable('W_att', [decoder_size, 1], initializer=xavier_initializer(), regularizer=regularizer)
                        b_att = tf.compat.v1.get_variable('b_att', [1], initializer=xavier_initializer())
                    with tf.compat.v1.variable_scope("velocity_enco"):
                        Wv_fw_enco = tf.compat.v1.get_variable('Wv_fw_enco', [decoder_size, decoder_size],
                                                               initializer=xavier_initializer(),
                                                               regularizer=regularizer)
                        Wv_bw_enco = tf.compat.v1.get_variable('Wv_bw_enco', [decoder_size, decoder_size],
                                                               initializer=xavier_initializer(),
                                                               regularizer=regularizer)


                def attention_lstm(decoder_feature, hidden_state, cell_state, attention_vector, attention_state, lstm):
                    attention_x = tf.matmul(decoder_feature, W)
                    attention_h = tf.matmul(hidden_state, W_h)
                    att = tf.nn.tanh(attention_x + tf.expand_dims(attention_h, axis=1))

                    # softmax attention weight  (batch, p, channel) -> (batch, p)
                    att = tf.reshape(att, [-1, decoder_size])
                    att = tf.add(tf.matmul(att, W_att), b_att)
                    att = tf.reshape(att, [-1, max_people])
                    att = tf.nn.softmax(att)
                    alpha = tf.expand_dims(att, axis=2)

                    # compute attention feature for lstm input
                    x = attention_x * alpha
                    attention_feature = tf.reduce_sum(x, axis=1)

                    # compute new state by attention feature and state
                    output, ((cell_state, hidden_state), attention_vector, attention_state) = lstm(attention_feature, ((cell_state, hidden_state), attention_vector, attention_state))

                    body_t = decoder_feature * alpha
                    body_t = tf.reduce_sum(body_t, axis=1)

                    return output, hidden_state, cell_state, attention_vector, attention_state, body_t, att

                mean_coords = tf.reduce_mean(fused_enco_coords, [1, 2])
                fw_hidden_state = tf.nn.tanh(tf.matmul(mean_coords, W_init_h_fw))
                fw_cell_state = tf.nn.tanh(tf.matmul(mean_coords, W_init_c_fw))
                bw_hidden_state = tf.nn.tanh(tf.matmul(mean_coords, W_init_h_bw))
                bw_cell_state = tf.nn.tanh(tf.matmul(mean_coords, W_init_c_bw))

                fw_attention_vector = tf.zeros([bs, lstm_size], tf.float32)
                fw_attention_state = tf.zeros([bs, attn_len, lstm_size], tf.float32)
                bw_attention_vector = tf.zeros([bs, lstm_size], tf.float32)
                bw_attention_state = tf.zeros([bs, attn_len, lstm_size], tf.float32)

                fw_body_seq = tf.TensorArray(dtype=tf.float32, size=max_frames, clear_after_read=False)
                bw_body_seq = tf.TensorArray(dtype=tf.float32, size=max_frames, clear_after_read=False)
                person_att = tf.TensorArray(dtype=tf.float32, size=max_frames)
                velocity_fw = tf.TensorArray(size=max_frames - 1, dtype=tf.float32)
                velocity_bw = tf.TensorArray(size=max_frames - 1, dtype=tf.float32)
                for t in range(max_frames):
                    with tf.compat.v1.variable_scope('body_fw_lstm'):
                        fw_output, fw_hidden_state, fw_cell_state, fw_attention_vector, fw_attention_state, fw_body, fw_att = \
                            attention_lstm(fused_enco_coords[:,t,:,:], fw_hidden_state, fw_cell_state, fw_attention_vector, fw_attention_state, body_fw_cell)
                        fw_body_seq = fw_body_seq.write(t, fw_body)
                        person_att = person_att.write(t, fw_att)
                    with tf.compat.v1.variable_scope('body_bw_lstm'):
                        bw_output, bw_hidden_state, bw_cell_state, bw_attention_vector, bw_attention_state, bw_body, _ = \
                            attention_lstm(fused_enco_coords[:, max_frames-1-t, :, :], bw_hidden_state, bw_cell_state, bw_attention_vector, bw_attention_state, body_bw_cell)
                        bw_body_seq = bw_body_seq.write(t, bw_body)

                    if t > 0:

                        velocity_fw_t = fw_body_seq.read(t) - fw_body_seq.read(t-1)


                        velocity_bw_t = bw_body_seq.read(t) - bw_body_seq.read(t-1)

                        velocity_fw = velocity_fw.write(t - 1, velocity_fw_t)
                        velocity_bw = velocity_bw.write(t - 1, velocity_bw_t)

                velocity_fw = velocity_fw.stack()  # t-1, bs, lstm_size
                velocity_bw = velocity_bw.stack()
                velocity_fw = tf.nn.tanh(tf.matmul(velocity_fw, Wv_fw_enco))
                velocity_bw = tf.nn.tanh(tf.matmul(velocity_bw, Wv_bw_enco))

                mean_velocity_fw = tf.reduce_mean(velocity_fw, 0)
                mean_velocity_bw = tf.reduce_mean(velocity_bw, 0)
                fw_v_hidden_state = tf.nn.tanh(tf.matmul(mean_velocity_fw, W_v_init_h_fw))
                fw_v_cell_state = tf.nn.tanh(tf.matmul(mean_velocity_fw, W_v_init_c_fw))
                bw_v_hidden_state = tf.nn.tanh(tf.matmul(mean_velocity_bw, W_v_init_h_bw))
                bw_v_cell_state = tf.nn.tanh(tf.matmul(mean_velocity_bw, W_v_init_c_bw))

                fw_v_attention_vector = tf.zeros([bs, lstm_size])
                fw_v_attention_state = tf.zeros([bs, attn_len, lstm_size])
                bw_v_attention_vector = tf.zeros([bs, lstm_size])
                bw_v_attention_state = tf.zeros([bs, attn_len, lstm_size])
                for t_v in range(max_frames - 1):
                    with tf.compat.v1.variable_scope('velocity_fw_lstm'):
                        fw_velocity_output, (
                        (fw_v_cell_state, fw_v_hidden_state), fw_v_attention_vector, fw_v_attention_state) = \
                            velocity_fw_cell(velocity_fw[t_v, :, :], (
                            (fw_v_cell_state, fw_v_hidden_state), fw_v_attention_vector, fw_v_attention_state))
                    with tf.compat.v1.variable_scope('velocity_bw_lstm'):
                        bw_velocity_output, (
                        (bw_v_cell_state, bw_v_hidden_state), bw_v_attention_vector, bw_v_attention_state) = \
                            velocity_bw_cell(velocity_bw[t_v, :, :], (
                            (bw_v_cell_state, bw_v_hidden_state), bw_v_attention_vector, bw_v_attention_state))
                velocity_out = tf.concat([fw_velocity_output, bw_velocity_output], -1)  # bs, lstm_size*2

                body_fw = tf.transpose(fw_body_seq.stack(), [1, 0, 2])
                body_bw = tf.transpose(bw_body_seq.stack(), [1, 0, 2])

                body_out = tf.concat([fw_output, bw_output], -1) #bs, lstm_size * 2
                body_out_seq = tf.concat([body_fw, body_bw], -1) #bs, t, lstm_size * 2

                with tf.compat.v1.variable_scope('cnn_attention'):
                    with tf.compat.v1.variable_scope("attention_cnn_x"):
                        W_c = tf.compat.v1.get_variable('W_c', [cnn_c, decoder_size], initializer=xavier_initializer(), regularizer=regularizer)
                    with tf.compat.v1.variable_scope("attention_cnn_h"):
                        W_c_h = tf.compat.v1.get_variable('W_c_h', [lstm_size*2, decoder_size], initializer=xavier_initializer(), regularizer=regularizer)
                    with tf.compat.v1.variable_scope("att_cnn"):
                        W_c_att = tf.compat.v1.get_variable('W_c_att', [decoder_size, 1], initializer=xavier_initializer(), regularizer=regularizer)
                        b_c_att = tf.compat.v1.get_variable('b_c_att', [1], initializer=xavier_initializer())

                fusion_att = tf.TensorArray(dtype=tf.float32, size=max_frames)
                mean_cnn = tf.reduce_mean(cnn_feature, [1, 2, 3])
                fu_hidden_state = tf.nn.tanh(tf.matmul(mean_cnn, W_init_h_fu))
                fu_cell_state = tf.nn.tanh(tf.matmul(mean_cnn, W_init_c_fu))
                fu_attention_vector = tf.zeros([bs, lstm_size], tf.float32)
                fu_attention_state = tf.zeros([bs, attn_len*lstm_size], tf.float32)

                def cnn_skel_attention(i, cnn_map, body_enco, fu_h, fu_c, fu_att_vec, fu_att_state, fusion_att, output):
                    cnn_i = cnn_map[:,i,:,:,:]
                    body_i = body_enco[:,i,:]
                    # compute pose-aware attention map
                    attention_cnn_x = tf.reshape(tf.matmul(cnn_i, W_c), [-1, cnn_h*cnn_w, decoder_size])
                    attention_cnn_h = tf.matmul(body_i, W_c_h)
                    att_cnn = tf.nn.tanh(attention_cnn_x + tf.expand_dims(attention_cnn_h, axis=1))
                    att_cnn = tf.add(tf.matmul(att_cnn, W_c_att), b_c_att)
                    att_cnn = tf.reshape(att_cnn, [-1, cnn_h * cnn_w])
                    alpha_cnn = tf.nn.softmax(att_cnn)  # (batch, 49)

                    x_cnn = attention_cnn_x * tf.expand_dims(alpha_cnn, axis=2)
                    attention_cnn_feature = tf.reduce_sum(x_cnn, axis=1)

                    output, ((fu_c, fu_h), fu_att_vec, fu_att_state) = fusion_cell(attention_cnn_feature, ((fu_c, fu_h), fu_att_vec, fu_att_state))
                    fusion_att = fusion_att.write(i, alpha_cnn)
                    return  i+1, cnn_map, body_enco, fu_h, fu_c, fu_att_vec, fu_att_state, fusion_att, output

                _, _, _, _, _, _, _, fusion_att, fu_output = tf.while_loop(cond=lambda i, *_: i<max_frames, body=cnn_skel_attention,
                                                                           loop_vars=(tf.constant(0, tf.int32), cnn_feature, body_out_seq,
                                                                                      fu_hidden_state, fu_cell_state, fu_attention_vector, fu_attention_state,
                                                                                      fusion_att, tf.zeros(dtype=tf.float32, shape=(bs, lstm_size))))

                fusion_attention_array = tf.transpose(fusion_att.stack(), [1, 0, 2])
                fusion_attention_array = tf.reshape(fusion_attention_array, [-1, max_frames, cnn_h, cnn_w])
                person_att_array = tf.transpose(person_att.stack(), [1,0,2])   #bs, max_frames, person


            return body_out, velocity_out, fu_output, fusion_attention_array, person_att_array


if __name__ == '__main__':
    cnn_map = tf.random_normal(shape=[3*16, 7, 7, 1024])
    skel_coords = tf.random_normal(shape= [3, 16, 4, 14, 2])
    body_out, fu_output, fusion_attention_array, _ = Part_Action_Rec(skel_coords, cnn_map)
    var_ls = tf.compat.v1.trainable_variables()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        body_out = sess.run(body_out)
        fu_output = sess.run(fu_output)
        att_array = sess.run(fusion_attention_array)
        print(body_out.shape)
        print(fu_output.shape)
        print(att_array.shape)
