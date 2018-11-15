import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

learning_rate = 0.01

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

train_accuracy_summary = tf.summary.scalar('acc_train', accuracy)
validation_accuracy_summary = tf.summary.scalar('acc_validation', accuracy)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 1000
batch_size = 50

acc_trains = []
acc_vals = []

with tf.Session() as sess:
    init.run()

    for epoch in range(n_epochs):
        X_batch, y_batch = mnist.train.next_batch(batch_size)
        sess.run(training_op, feed_dict={X:X_batch, y:y_batch})

        if epoch%10 == 0:
            train_summary_str = train_accuracy_summary.eval(feed_dict={X: X_batch, y:y_batch})
            validation_summary_str = validation_accuracy_summary.eval(feed_dict={X: X_batch, y:y_batch}) 
            file_writer.add_summary(train_summary_str, epoch)
            file_writer.add_summary(validation_summary_str, epoch)

            acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
            acc_val = accuracy.eval(feed_dict={X:mnist.validation.images,
                                           y:mnist.validation.labels})
            print(epoch, "Train accuracy:", acc_train, "Val accuracy:", acc_val)
    save_path = saver.save(sess, "./model/my_model_final.ckpt")
