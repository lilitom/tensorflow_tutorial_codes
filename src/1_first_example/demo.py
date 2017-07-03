# 这是tensorflow实现的线性回归例子

import tensorflow as tf

# 准备数据
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# 定义好模型
W = tf.Variable(3., dtype=tf.float32)
b = tf.Variable(-3., dtype=tf.float32)

x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)

y_ = tf.multiply(x, W) + b  # 我们的线性模型

# 定义loss function
loss = tf.reduce_mean(tf.square(y - y_))

# 创建优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 迭代模型
sess = tf.Session()
sess.run(tf.global_variables_initializer())

tf.summary.FileWriter(logdir='graphs', graph=sess.graph)

for _ in range(1000):
    sess.run(train, feed_dict={x: x_train, y: y_train})

# 查看模型结果
cur_W, cur_b, cur_loss = sess.run([W, b, loss], feed_dict={x: x_train, y: y_train})
print('W: %s, b: %s, loss: %s' % (cur_W, cur_b, cur_loss))

sess.close()
