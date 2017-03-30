import tensorflow as tf


X = tf.Variable(tf.constant(5., dtype=tf.float32))
y = tf.square(X)
iterations = 100


# 使用大的学习率
# train_step = tf.train.GradientDescentOptimizer(1).minimize(y)
# 结果振荡
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for step in range(iterations):
#         sess.run(train_step)
#         if step % 5 == 0:
#             print('After %d training step(s) , x is' % step)
#             print(sess.run(X))

# 使用小的学习率
# 最后结果大于0.7
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(y)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for step in range(iterations):
#         sess.run(train_step)
#         if step % 5 == 0:
#             print('After %d training step(s) , x is' % step)
#             print(sess.run(X))


# 使用指数衰减
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.1, global_step, 1, 0.96, staircase=True)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(y)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(iterations):
        sess.run(train_step)
        if step % 5 == 0:
            print('After %d training step(s) , x is' % step)
            print(sess.run(X))
