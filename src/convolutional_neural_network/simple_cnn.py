from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def weight_variables(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variables(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


mnist = input_data.read_data_sets('mnist_data', one_hot=True)

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])
x_image = tf.reshape(x, shape=[-1, 28, 28, 1])

w_conv1 = weight_variables([5, 5, 1, 32])
b_conv1 = bias_variables([32])
h_conv1 = tf.nn.relu(tf.add(conv2d(x_image, w_conv1), b_conv1))
h_pool1 = max_pool_2x2(h_conv1)

print(tf.shape(h_pool1))
