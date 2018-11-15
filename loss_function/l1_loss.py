import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1., 1., 500)
y = np.zeros(500)

target = tf.placeholder(tf.float32, [500])
predict = tf.placeholder(tf.float32, [500])

l1_loss = tf.abs(target-predict)

with tf.Session() as sess:
  l1_output = sess.run(l1_loss, feed_dict={target: y, predict: x})

plt.plot(x, l1_output, 'b--', label='L1 LOSS')
plt.show()
