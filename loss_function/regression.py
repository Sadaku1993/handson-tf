import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1., 1., 500)
y = np.zeros(500)

target = tf.placeholder(tf.float32, [500])
predict = tf.placeholder(tf.float32, [500])

l1_loss = tf.abs(target-predict)
l2_loss = tf.square(target-predict)

def huber_loss(prediction, label, delta=0.5):
    error = label - prediction
    cond = tf.abs(error) < delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * ( tf.abs(error) - 0.5 * delta)
    return tf.where(cond, squared_loss, linear_loss)
  
with tf.Session() as sess:
  l1_output = sess.run(l1_loss, feed_dict={target: y, predict: x})
  l2_output = sess.run(l2_loss, feed_dict={target: y, predict: x})
  huber_output = sess.run(huber_loss(x, y))


plt.plot(x, l1_output, 'b--', label='L1 LOSS')
plt.plot(x, l2_output, 'r--', label='L2 LOSS')
plt.plot(x, huber_output, 'g--', label='Huber LOSS')
plt.show()
