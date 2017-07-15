from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

import tensorflow as tf
import numpy as np
import random


def build_data(n):
    xs = []
    ys = []
    for i in range(0, 2000):
        k = random.uniform(1, 50)

        x = [[np.sin(k + j)] for j in range(0, n)]
        y = [np.sin(k)]

        xs.append(x)
        ys.append(y)

    train_x = np.array(xs[:1500])
    test_x = np.array(xs[1500:])
    train_y = np.array(ys[:1500])
    test_y = np.array(ys[1500:])
    return train_x, train_y, test_x, test_y


length = 10
time_step_size = length
vector_size = 1
batch_size = 10
test_size = 10

train_x, train_y, test_x, test_y = build_data(length)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

x = tf.placeholder(dtype=tf.float32, shape=[None, length, vector_size])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([10, 1], stddev=0.01))
b = tf.Variable(tf.random_normal([1], stddev=0.01))


def seq_predict_model(x, w, b, time_step_size, vector_size):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, vector_size])
    x = tf.split(x, time_step_size, 0)

    cell = core_rnn_cell.BasicRNNCell(num_units=10)
    initial_state = tf.zeros([batch_size, cell.state_size])
    outputs, _states = core_rnn.static_rnn(cell, x, initial_state=initial_state)

    return tf.matmul(outputs[-1], w) + b, cell.state_size


y, _ = seq_predict_model(x, w, b, time_step_size, vector_size)
loss = tf.square(tf.subtract(y_, y))
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(50):

        for end in range(batch_size, len(train_x), batch_size):
            begin = end - batch_size
            x_value = train_x[begin: end]
            y_value = train_y[begin: end]
            sess.run(train_op, feed_dict={x: x_value, y_: y_value})

        test_indices = np.arange(len(test_x))
        np.random.shuffle(test_indices)
        test_indices = test_indices[: test_size]
        x_value = test_x[test_indices]
        y_value = test_y[test_indices]

        val_loss = np.mean(sess.run(loss, feed_dict={x: x_value, y_: y_value}))
        print('Run %d , loss %f' % (i, val_loss))
