import tensorflow as tf
import numpy as np

# 生成数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 构建网络
x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 1])


def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    outputs = tf.matmul(inputs, weights) + biases
    if activation_function is not None:
        outputs = activation_function(outputs)
    return outputs


h1 = add_layer(x, 1, 20, activation_function=tf.nn.relu)
y = add_layer(h1, 20, 1, activation_function=None)


# loss = tf.reduce_mean(tf.squared_difference(y, y_))
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), axis=1))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1001):
    sess.run(train_step, feed_dict={x: x_data, y_: y_data})
    if i % 50 == 0:
        tloss = sess.run(loss, feed_dict={x: x_data, y_: y_data})
        print('step: {}, loss: {}'.format((i + 1), tloss))

sess.close()
