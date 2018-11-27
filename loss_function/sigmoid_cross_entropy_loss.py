import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3., 5., 500)
y = np.ones(500)
y_ = np.zeros(500)

predict = tf.placeholder(tf.float32, [500])
target = tf.placeholder(tf.float32, [500])

sigmoid_cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=predict, labels=target)

with tf.Session() as sess:
    loss = sess.run(sigmoid_cross_entropy_loss, feed_dict={predict: x, target: y})
    loss_ = sess.run(sigmoid_cross_entropy_loss, feed_dict={predict: x, target: y_})

plt.plot(x, loss_, 'r--', label="SIGMOID CROSS ENTROPY LOSS (label=0)")
plt.plot(x, loss, 'g--', label="SIGMOID CROSS ENTROPY LOSS (label=1)")
plt.legend(loc='lower right', prop={'size': 11})
plt.show()
