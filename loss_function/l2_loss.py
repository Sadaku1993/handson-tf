import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1., 1., 500)
y = np.zeros(500)

target = tf.placeholder(tf.float32, [500])
predict = tf.placeholder(tf.float32, [500])

l2_loss = tf.square(target-predict)

with tf.Session() as sess:
  l2_output = sess.run(l2_loss, feed_dict={target: y, predict: x})

plt.plot(x, l2_output, 'b--', label='L2 LOSS')
plt.show()
