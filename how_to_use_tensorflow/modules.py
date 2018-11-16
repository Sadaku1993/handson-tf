#coding:utf-8
import tensorflow as tf

def relu(X):
    with tf.name_scope("relu") as scope:
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, 0.0, name="relu")

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")

file_writer = tf.summary.FileWriter("logs/relu1", tf.get_default_graph())

init = tf.global_variables_initializer() # initノードを準備

with tf.Session() as sess:
    sess.run(init)
    result = output.eval(feed_dict={X:[[1,2,3]]})
print(result)
