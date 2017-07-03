"""
Simple linear regression example in TensorFlow
"""

import tensorflow as tf

# data preparation
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# difine the model
W = tf.Variable(3., dtype=tf.float32)
b = tf.Variable(-3., dtype=tf.float32)

x = tf.placeholder(dtype=tf.float32)
y_ = tf.placeholder(dtype=tf.float32)

y = tf.multiply(x, W) + b  # 我们的线性模型

# difine the loss function
loss = tf.reduce_mean(tf.square(y_ - y))

# create optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# iterative model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# tensorboard --logdir "./graphs"
# tf.summary.FileWriter(logdir='graphs', graph=sess.graph)

for _ in range(1000):
    sess.run(train, feed_dict={x: x_train, y_: y_train})

# check the model results
cur_W, cur_b, cur_loss = sess.run([W, b, loss], feed_dict={x: x_train, y_: y_train})
print('W: %s, b: %s, loss: %s' % (cur_W, cur_b, cur_loss))

sess.close()
