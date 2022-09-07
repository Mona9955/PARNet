import tensorflow as tf
import numpy as np
import random


def trans(x,y):
    c = x
    d = y * random.randint(1,3)
    return c, d

x = np.arange(13)
y = np.arange(6,19)
bs = 8
dataset = tf.data.Dataset.from_tensor_slices({'x':tf.convert_to_tensor(x),'y':tf.convert_to_tensor(y)})
dataset = dataset.shuffle(buffer_size = len(x)//2)
d1 = dataset.map(lambda d: trans(d['x'], d['y']), num_parallel_calls=2)
d1 = d1.batch(bs)
d1 = d1.prefetch(buffer_size = bs)
iterator = tf.data.Iterator.from_structure(d1.output_types, d1.output_shapes)
# iterator = tf.compat.v1.data.make_one_shot_iterator(d1)
# iterator = tf.compat.v1.data.make_initializable_iterator(d1)
# iterator_init = iterator.make_initializer(d1)
z, k = iterator.get_next()
with tf.compat.v1.Session() as sess:
    # sess.run(iterator_init)
    # count = int(np.ceil(len(x)/bs))
    count = 2
    # sess.run(iterator.initializer)
    for j in range(3):
        # b = sess.run(k)
        # print(b)
        sess.run(iterator.make_initializer(d1))
        while True:
            try:
                b = sess.run(k)

                print(b)

            except tf.errors.OutOfRangeError:

                print('*'*10)
                break
