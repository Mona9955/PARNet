import tensorflow as tf
import numpy as np
import config_2d as cfg
from tqdm import tqdm
from Model_skel_cnn_vel import Act_2D_Model
from dataloader_2d_skeleton_multi_thread import *
from dataloader_all import *
import time
import os
import collections

mark = f'ucf101_sp1'
num_gpus = len(cfg.GPUS)
image_size = cfg.IMG_SIZE
total_epochs = cfg.NUM_EPOCHS
lr_init = cfg.LR_RATE
max_frames = 16
max_size = 256
bs_train = cfg.BATCH_SIZE * num_gpus
bs_test = cfg.BATCH_SIZE_TEST
warm_up_steps = cfg.WARM_UP_STEPS

max_people = cfg.MAX_PEOPLE
max_points = cfg.MAX_POINTS
action_label = list(cfg.LABEL_DICT.keys())
num_classes = len(action_label)


os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in cfg.GPUS])

def cal_pred(input):
    probs = tf.nn.softmax(input, axis=-1)
    pred_class = tf.argmax(probs, axis=-1)
    return pred_class, probs


def average_gradients(tower_grads):

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            try:
                expend_g = tf.expand_dims(g, 0)
                grads.append(expend_g)

            except Exception:
                print(g, _)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


class Batch_Iter(object):
    def __init__(self, input, bs=bs_train):
        self.total_num = len(input[0])
        self.iter_num = self.total_num // bs
        self.last_batch_num = self.total_num % bs
        self.input = input
        self.bs = bs
        self.count = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.iter_num and self.last_batch_num != 0:
            data_batch = [item[self.count * self.bs: self.total_num] for item in self.input]
        elif self.count < self.iter_num:
            data_batch = [item[self.count * self.bs: (self.count + 1) * self.bs] for item in self.input]
        else:
            raise StopIteration()
        return data_batch


def warm_up_lr(step, factor=0.01, milestone=cfg.WARM_UP_STEPS, init_lr = lr_init*0.5):
    alpha = step / milestone
    warm_up_factor = factor * (1-alpha) + alpha
    out_lr = tf.cast(warm_up_factor * init_lr, tf.float32)
    return out_lr

if __name__ == '__main__':


    with tf.Graph().as_default():

        # _________________________________________train_________________________________________________

        image_holder = tf.compat.v1.placeholder(tf.float32, [None, max_frames, image_size, image_size, 3], name='image')
        coords_holder = tf.compat.v1.placeholder(tf.float32, [None, max_frames, max_people, max_points, 2], name='coords')
        label_holder = tf.compat.v1.placeholder(tf.int32, [None], name='label')

        label_holder_split = tf.split(label_holder, num_or_size_splits=num_gpus, axis=0)
        coords_holder_split =  tf.split(coords_holder, num_or_size_splits=num_gpus, axis=0)
        image_holder_split = tf.split(image_holder, num_or_size_splits=num_gpus, axis=0)

        global_step = tf.Variable(1, trainable=False, name='global_step')
        lr_list = [lr_init, lr_init*0.5, lr_init * 0.25, lr_init * 0.1]
        lr = tf.cond(global_step < warm_up_steps, lambda: warm_up_lr(global_step, init_lr=lr_list[0]),
                     lambda: tf.compat.v1.train.piecewise_constant_decay(global_step, boundaries=[78000, 128000, 138000], values=lr_list))  #[28000, 38000, 42000] [20000, 28000, 30000]

                     # lambda: tf.compat.v1.train.exponential_decay(lr_init, global_step, 28000, 0.5, staircase=True))

        optimizer = tf.compat.v1.train.MomentumOptimizer(lr, momentum=0.9)
        tower_grad = []
        if_reuse=None

        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.compat.v1.name_scope('tower_%d' % i) as scope:
                    label_holder_split_i = label_holder_split[i]
                    coords_holder_split_i = coords_holder_split[i]
                    image_holder_split_i = image_holder_split[i]
                    output_all, output_cnn, output_body, output_velocity, output_fu, fusion_attention_array = Act_2D_Model(image_holder_split_i, coords_holder_split_i, num_classes=num_classes, name='Act_2D', is_training=True, reuse=if_reuse, dropout_keep_prob=0.5, cell_dropout_rate=0.2)
                    pred_class, _ = cal_pred(output_all)

                    one_hot_label_batch = tf.one_hot(label_holder_split_i, num_classes)

                    loss_all = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_all, labels=one_hot_label_batch))
                    loss_cnn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_cnn, labels=one_hot_label_batch))
                    loss_body = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_body, labels=one_hot_label_batch))
                    loss_velocity = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_velocity, labels=one_hot_label_batch))
                    loss_fu = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_fu, labels=one_hot_label_batch))

                    l2_loss = tf.add_n(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
                    total_loss = loss_all + loss_cnn + loss_body + loss_velocity + loss_fu + l2_loss
                    # if i == 0:
                    #     update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, scope=scope)
                    grad = optimizer.compute_gradients(total_loss)
                    tower_grad.append(grad)
                    if_reuse = True

        grads = average_gradients(tower_grad)
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            apply_avg_grads_op = optimizer.apply_gradients(grads, global_step)
        # __________________________________________________________________________________________________
        # ___________________________________________test___________________________________________________

        test_image_holder = tf.compat.v1.placeholder(tf.float32, [None, max_frames, image_size, image_size, 3], name='image_test')
        test_coords_holder = tf.compat.v1.placeholder(tf.float32, [None, max_frames, max_people, max_points, 2], name='coords_test')
        test_label_holder = tf.compat.v1.placeholder(tf.int32, [None], name='label_test')
        with tf.device('/gpu:0'):
            with tf.compat.v1.name_scope('tower_0') as scope:
                output_all_t, output_cnn_t, output_body_t, output_velocity_t, output_fu_t, _ = Act_2D_Model(test_image_holder, test_coords_holder, num_classes=num_classes, is_training=False, reuse=True, dropout_keep_prob=1., cell_dropout_rate=0.)
                pred_all, _ = cal_pred(output_all_t)
                pred_cnn, score_cnn = cal_pred(output_cnn_t)
                pred_body, score_body = cal_pred(output_body_t)
                pred_velocity, score_velocity = cal_pred(output_velocity_t)
                pred_fu, score_fu = cal_pred(output_fu_t)

                pred_add = tf.argmax(score_fu+score_body+score_velocity+score_cnn, axis=-1)

                one_hot_label_batch_test = tf.one_hot(test_label_holder, num_classes)
                loss_test = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_all_t, labels=one_hot_label_batch_test))
                l2_loss_test = tf.add_n(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
                total_loss_test = loss_test + l2_loss_test
        # __________________________________________________________________________________________________



        # incre_global_step = tf.compat.v1.assign(global_step, global_step + 1)
        # train_op = tf.group(optimizer, incre_global_step)
        # train_op = apply_avg_grads_op
        tf.compat.v1.summary.image(name='input_image', tensor=image_holder_split_i[0,:,:,:,:], max_outputs=max_frames)
        tf.compat.v1.summary.image(name='attention', tensor=tf.expand_dims(fusion_attention_array[0,:,:,:], axis=-1), max_outputs=max_frames)
        tf.compat.v1.summary.scalar(name='learning_rate', tensor=lr)
        tf.compat.v1.summary.scalar(name='concat_all_loss', tensor=loss_all)
        tf.compat.v1.summary.scalar(name='body_loss', tensor=loss_body)
        tf.compat.v1.summary.scalar(name='velocity_loss', tensor=loss_velocity)
        tf.compat.v1.summary.scalar(name='cnn_loss', tensor=loss_cnn)
        tf.compat.v1.summary.scalar(name='fusion_loss', tensor=loss_fu)
        tf.compat.v1.summary.scalar(name='l2_loss', tensor=l2_loss)
        tf.compat.v1.summary.scalar(name='total_loss', tensor=total_loss)


        tf.compat.v1.summary.tensor_summary(name='train_pred', tensor=pred_class)
        tf.compat.v1.summary.text(name='train_pred', tensor=tf.as_string(pred_class))
        tf.compat.v1.summary.tensor_summary(name='label', tensor=label_holder_split_i)
        tf.compat.v1.summary.text(name='label', tensor=tf.as_string(label_holder_split_i))

        merge_summary_op = tf.compat.v1.summary.merge_all()

        var_list = tf.compat.v1.trainable_variables()
        global_list = tf.compat.v1.global_variables()
        bn_moving_vars = [g for g in global_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in global_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        saver = tf.compat.v1.train.Saver(var_list=var_list, max_to_keep = 10)

        if not os.path.exists(cfg.MODEL_DIR):
            os.makedirs(cfg.MODEL_DIR)
        train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        model_name = f'{mark}_rgb_skel_model_{train_start_time}'
        model_save_path = os.path.join(cfg.MODEL_DIR, model_name)
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        test_dict = test_data_loader_101()
        image_dir_test, skel_dir_test, label_test = test_dict['image_path'], test_dict['skel_path'], test_dict['label']

        sess_config=tf.compat.v1.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        var_restore = tf.compat.v1.trainable_variables(scope='InceptionV2') #scope='InceptionV2'
        # var_restore = [var for var in var_restore if "beta" not in var.name]
        # var_classify = tf.compat.v1.trainable_variables(scope='final_output_class')
        # var_restore = [var for var in var_list if 'class' not in var.op.name]
        loader = tf.compat.v1.train.Saver(var_list=var_restore)

        with tf.compat.v1.Session(config=sess_config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            summary_writer = tf.compat.v1.summary.FileWriter(model_save_path)
            summary_writer.add_graph(sess.graph)

            reload = True

            pre_dir = cfg.PRETRAINED_DIR

            if reload and cfg.PRETRAINED_DIR is not None:
                print('restoring model ckpt...')
                loader.restore(sess, pre_dir)   #cfg.PRETRAINED_DIR
                print('restore finished!')
            best_accuracy = 0
            for epoch in range(1, total_epochs+1):
                train_dict = train_data_loader_101()
                image_dir_train, skel_dir_train, label_train = train_dict['image_path'], train_dict['skel_path'], train_dict['label']
                data_batch = Batch_Iter([image_dir_train, skel_dir_train, label_train], bs=bs_train)
                # count = 0
                for data in data_batch:
                    # count += 1
                    image_dir_data, skel_dir_data, label_data = data
                    if len(label_data) % num_gpus != 0:
                        continue

                    step = global_step.eval(session=sess)
                    if not cfg.USE_MT:
                        img_data, skel_data = process_batch_train(image_dir_data, skel_dir_data)
                    else:
                        img_data, skel_data = process_batch_mt(image_dir_data, skel_dir_data, process_batch_train)

                    label_data = np.array(label_data, dtype=np.int32)
                    _, learning_rate, l_concat, l_cnn, l_skel, l_vel, l_fu, total_loss_1, summary = sess.run([apply_avg_grads_op, lr, loss_all, loss_cnn, loss_body, loss_velocity, loss_fu, total_loss, merge_summary_op],
                                                                              feed_dict={image_holder:img_data, coords_holder:skel_data, label_holder:label_data})

                    line_train = "epoch: %d, global_step: %d, learning_rate: %.8f, loss_total: %.2f, loss_concat: %.2f, loss_skel: %.2f, loss_vel: %.2f, loss_fu: %.2f, loss_cnn: %.2f" \
                                 % (epoch, step, learning_rate, total_loss_1, l_concat, l_skel, l_vel, l_fu, l_cnn)
                    print('\r'+line_train, end=' ')

                    if step % cfg.SUMMARY_PERIOD == 0:
                        summary_writer.add_summary(summary=summary, global_step=step)

                    # if step % cfg.SAVE_PERIOD == 0:
                    #     saver.save(sess=sess, save_path=model_save_path + '/model', global_step=step)
                    #-------------test_period-------------------

                    test_period = 200

                    if step % test_period == 0:
                        print('\nStart evaluation......')

                        test_loss_total = 0
                        test_acc_concat = 0
                        test_acc_skel = 0
                        test_acc_vel = 0
                        test_acc_fu = 0
                        test_acc_cnn = 0
                        test_acc_add = 0

                        data_num = 0
                        test_num = len(label_test)
                        test_batch_num =  test_num // bs_test
                        if len(label_test) % bs_test != 0:
                            test_batch_num += 1
                        wrong_dict = {}
                        with tf.device('/gpu:0'):
                            with tqdm(total=test_batch_num) as pbar:
                                for data_test in Batch_Iter([image_dir_test, skel_dir_test, label_test], bs=bs_test):
                                    image_dir_data_test, skel_dir_data_test, label_data_test = data_test

                                    img_data_test, skel_data_test = process_batch_test(image_dir_data_test, skel_dir_data_test, max_size=max_size)
                                    label_data_test = np.array(label_data_test, dtype=np.int32)

                                    pred_concat_test, pred_body_test, pred_velocity_test, pred_fu_test, pred_cnn_test, l_test, pred_add_test = sess.run([pred_all, pred_body, pred_velocity, pred_fu, pred_cnn, total_loss_test, pred_add],
                                                                                                 feed_dict={test_image_holder:img_data_test, test_coords_holder:skel_data_test, test_label_holder:label_data_test})

                                    test_acc_concat += np.sum(pred_concat_test == label_data_test)
                                    test_acc_skel += np.sum(pred_body_test == label_data_test)
                                    test_acc_vel += np.sum(pred_velocity_test == label_data_test)
                                    test_acc_fu += np.sum(pred_fu_test == label_data_test)
                                    test_acc_cnn += np.sum(pred_cnn_test == label_data_test)
                                    test_acc_add += np.sum(pred_add_test == label_data_test)
                                    #----------------------------------------------------
                                    for i in range(len(label_data_test)):
                                        gt_action = action_label[int(label_data_test[i])]
                                        if pred_concat_test[i] != label_data_test[i]:
                                            if gt_action not in wrong_dict.keys():
                                                wrong_dict[gt_action] = 1
                                            else:
                                                wrong_dict[gt_action] += 1
                                    # ----------------------------------------------------
                                    test_loss_total += l_test
                                    pbar.update(1)

                        acc_con_f = round(test_acc_concat*100./test_num, 2)
                        acc_add_f = round(test_acc_add*100./test_num, 2)

                        with open(model_save_path+'/test_result.txt', 'a', encoding='utf-8') as f:
                            acc_total = max(acc_con_f, acc_add_f)
                            if acc_total > best_accuracy:
                                best_accuracy = acc_total
                                saver.save(sess=sess, save_path=model_save_path + f'/model{acc_total}', global_step=step)


                            line = "epoch:%d, global_step:%d, learning_rate:%.5f, test_loss:%.2f, acc_concat:%.2f, acc_add:%.2f, acc_skel:%.2f, acc_vel:%.2f, acc_fu:%.2f, acc_cnn:%.2f" \
                                   %(epoch, step, learning_rate, test_loss_total/test_batch_num, acc_con_f, acc_add_f, test_acc_skel*100./test_num, test_acc_vel*100./test_num,
                                     test_acc_fu*100./test_num, test_acc_cnn*100./test_num)

                            f.write(line + '\n')
                            print('Test result......')
                            print(line)
                            print(wrong_dict)
            summary_writer.close()

