from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# print('training data:', mnist.train.images.shape, )
# print('testing data', mnist.test.images.shape)
# print('validation data', mnist.validation.images.shape)

x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]), dtype=tf.float32)
b = tf.Variable(tf.zeros(10), dtype=tf.float32)

y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(128)
    sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    if (i + 1) % 50 == 0:
        loss = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
        print('Epoch: {0}, currect loss: {1}'.format(i + 1, loss))

correct_prediction = tf.equal(tf.argmax(y_, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
model_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

# model accuracy: 0.921
print('model accuracy: {0}'.format(model_acc))


sess.close()