import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1., 1., 500)
y = np.zeros(500)

def huber_loss(prediction, label, delta=0.5):
    error = label - prediction
    cond = tf.abs(error) < delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * ( tf.abs(error) - 0.5 * delta)
    return tf.where(cond, squared_loss, linear_loss)

with tf.Session() as sess:
    huber_output = sess.run(huber_loss(x, y))
plt.plot(x, huber_output, 'g--', label='Huber LOSS')
plt.show()
