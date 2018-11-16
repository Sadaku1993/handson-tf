import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 5, 500)
y = np.ones(500)

target = tf.placeholder(tf.float32, [500])
predict = tf.placeholder(tf.float32, [500])

xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(logits=predict, labels=target)

with tf.Session() as sess:
    loss = sess.run(xentropy_sigmoid_y_vals, feed_dict={predict: y, target: x})

plt.plot(x, loss, 'r--', label="CROSS ENTROPY LOSS")
plt.show()
