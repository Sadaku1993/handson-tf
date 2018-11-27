import tensorflow as tf

outputs = tf.nn.sigmoid(0.4)
action_prob = tf.concat([outputs, 1-outputs], 0)
action = tf.multinomial(tf.log(action_prob), num_samples=1)

with tf.Session() as sess:
    outputs_ = sess.run(outputs)
    action_prob_ = sess.run(action_prob)
    action_ = sess.run(action)

print(outputs_)
print(action_prob_)
print(action_)
