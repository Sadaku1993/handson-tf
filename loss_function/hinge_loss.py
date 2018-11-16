import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3., 5., 500)
y = np.ones(500)

target = tf.placeholder(tf.float32, [500])
predict = tf.placeholder(tf.float32, [500])

hinge_loss = tf.maximum(0., 1 - tf.multiply(target, predict))

with tf.Session() as sess:
    hinge_output = sess.run(hinge_loss, feed_dict={target: y, predict: x})

plt.plot(x, hinge_output, 'b--', label='HINGE LOSS')
plt.show()
