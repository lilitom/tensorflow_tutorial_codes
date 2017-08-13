import numpy as np
import tensorflow as tf


def xavier_init(n_in, n_out, factor=1.):
    low = -factor * np.sqrt(6. / (n_in + n_out))
    high = factor * np.sqrt(6. / (n_in + n_out))
    return tf.random_uniform(shape=[n_in, n_out], minval=low,
                             maxval=high, dtype=tf.float32)


class AdditiveGaussianNoiseAutoencoder():
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(),
                 scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weight()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal(n_input),
                                                     self.weights['w1']), self.weights['b1']))
        self.reconstrunction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x, self.reconstrunction), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weight(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run([self.cost, self.optimizer], feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    def calc_total_cost(self, X):
        cost = self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

