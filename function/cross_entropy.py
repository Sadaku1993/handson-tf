#coding:utf-8
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 3])
label = tf.placeholder(tf.float32, shape=[None, 3])
sparse_label = tf.placeholder(tf.int32, shape=[None])

# 個別に計算
softmax = tf.nn.softmax(x)
y1 = -tf.reduce_sum(tf.log(softmax) * label, axis=1)

# softmax_cross_entropy_with_logits
# ソフトマックスとクロスエントロピを同時に実行してくれる
# labelは確率分布(one hotである必要はない)
y2 = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=label)

# softmax_cross_entropy_with_logits
# ソフトマックスとウロスエントロピを同時に実行してくれる
# labelは分類先のインデックスを指定する
y3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=sparse_label)

sess = tf.Session()

x_ = [[6, 5, 4],
      [2, 5, 4],
      [3, 1, 6],
      [3, 1, 6]]
l = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1],
     [0.3, 0.1, 0.6]]
sl = [0, 1, 2, 2]

y1_ = sess.run([y1], feed_dict={x:x_, label:l})
y2_ = sess.run([y2], feed_dict={x:x_, label:l})
y3_ = sess.run([y3], feed_dict={x:x_, sparse_label:sl})

print("y1_:", y1_)

print("y2_:", y2_)

print("y3_:", y3_)
