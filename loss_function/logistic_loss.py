import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3., 5., 500)
y = np.ones(500)
y_ = np.zeros(500)

target = tf.placeholder(tf.float32, [500])
predict = tf.placeholder(tf.float32, [500])

entropy_loss = -tf.multiply(target, tf.log(predict)) - \
                    tf.multiply((1. - target), tf.log(1. - predict))

with tf.Session() as sess:
    entropy_output = sess.run(entropy_loss, feed_dict={target:y, predict:x})
    entropy_output_ = sess.run(entropy_loss, feed_dict={target:y_, predict:x})

plt.plot(x, entropy_output, 'g--', label='CROSS ENTROPY LOSS (label=1)')
plt.plot(x, entropy_output_, 'r--', label='CROSS ENTROPY LOSS (label=0)')
plt.show()
