import tensorflow as tf
import numpy as np

samples = tf.multinomial(tf.log([[0.2, 0.8]]), num_samples=1)

count = 0

with tf.Session() as sess:

    for i in range(100):
        output = samples.eval()

        if output:
            count+=1

print(count)
