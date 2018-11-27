import tensorflow as tf
import matplotlib.pyplot as plt

x_vals = tf.linspace(-3., 5., 500)
targets = tf.fill([500], 1.)

x_val_input = tf.expand_dims(x_vals, 1)
target_input = tf.expand_dims(targets, 1)

vals = tf.nn.sigmoid_cross_entropy_with_logits(
        logits = x_val_input, labels=target_input)

with tf.Session() as sess:
    out = sess.run(vals)
    xarray = sess.run(x_vals)

plt.plot(xarray, out, 'b--', label='loss')
plt.show()
