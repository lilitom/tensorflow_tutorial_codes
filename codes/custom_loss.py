import tensorflow as tf
import numpy as np

# 构建计算图
x_input = tf.placeholder(dtype=tf.float32, shape=[None, 2])
y_output = tf.placeholder(dtype=tf.float32, shape=[None, 1])
weight = tf.Variable(tf.random_normal(shape=[2, 1], stddev=1.))
y_pred = tf.matmul(x_input, weight)

# 自定义损失函数
# y_pred > weight 损失权重为 10
# y_pred <= weight 损失权重为 1


less_loss = 1.
more_loss = 10.
loss = tf.reduce_sum(tf.where(tf.greater(y_output, y_pred),
                              less_loss * (y_output - y_pred), more_loss * (y_pred - y_output)))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# rand一些数据
# 加入noise非常重要!!!
X = np.random.rand(128, 2)
y = np.array([[x1 + x2 + np.random.rand() / 10. - 0.05] for x1, x2 in X])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_size = 8
    steps = 5001
    for s in range(steps):
        rand_indics = np.random.choice(X.shape[0], batch_size)
        rand_X = X[rand_indics]
        rand_y = y[rand_indics]
        sess.run(train_step, feed_dict={x_input: rand_X, y_output: rand_y})
        if s % 200 == 0:
            print('After %d training step(s), weight is' % s)
            print(sess.run(weight))


