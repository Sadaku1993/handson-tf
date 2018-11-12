[Hands-on Machine Learning with Scikit-Learn and TensorFlow](https://github.com/ageron/handson-ml)の第９章のまとめ。
この本の第1章~第4章までを理解出来ている方が望ましいです。


[![book](http://akamaicovers.oreilly.com/images/0636920052289/cat.gif)](http://shop.oreilly.com/product/0636920052289.do)

# グラフ作成をセッション内での実行

まずは簡単な計算$f(x,y)=X*x*y+y+2$の計算方法についてまとめる。

```python
import tensorflow as tf

# 計算グラフを作成
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

# セッションを開き、変数を初期化してfをセッションを閉じる
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()
```

注意すべきなのは計算グラフだけではコードは何もしない点です。
計算グラフを評価するためには、Tensorflow セッションを開き、変数を初期化し、fを評価する必要があります。

sess.run()を毎回行うのは面倒なので、with tf.Session() as sessを利用することで簡略化することができます。

```python
import tensorfow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*y*y + y + 2

with tf.Session() as sess
  x.initializer.run()
  y.initializer.run()
  result = f.eval()
  print(result)
```

また、一つ一つの変数について初期化をそれぞれ書くのも面倒なのでglobal_variables_initializer()関数により簡略化できます。

```python
import tensorflow as tf

x=tf.Variable(3, name="x")
y=tf.Variable(4, name="y")
f = x*x*y + y + 2

init = tf.global_variables_initializer() # initノードを準備

with tf.Session as sess:
  init.run() # すべての変数を初期化する
　result = f.eval()
  print(result)
```

# Tensorflowによる線形回帰
カルフォルニアの住宅価格データセットに対する線形回帰を行う。
属性は、MedInc(収入の中央値), HouseAge(築年数), AveRooms(部屋数の中央値), AveBedrms(寝室の中央値), Population(人口), AveOccup(海との位置関係の中央値), Latitude(緯度), Longitude(経度)である。
これらの属性より、AgeHouses(住宅価格の中央値)を算出する線形回帰モデルを作成します。

正規方程式は

```math
\theta = (X^T \cdot X)^{-1} \cdot X^T \cdot y

```
にて表されます。

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data] # バイアス入力特徴量を追加

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="Y") # 行ベクトルに変換

XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as tf:
    theta_value = theta.eval()
    print(theta_value)
```


# 勾配降下法の実装
バッチ勾配降下法をtensorflowで実装します。

## マニュアルの勾配計算

- tf.assign：assign()関数は、変数に新しい値を代入するノードを作る。バッチ勾配降下法のステップ$\theta^{next\ step}=\theta-\eta\delta_\theta MSE(\theta)$に利用する。

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

n_epochs = 1000
learning_rate = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="Y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta-learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch%100 == 0:
            print("Epoch", epoch, " MSE=", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()
    print(best_theta)
```

## 自動微分による勾配計算

線形回帰のようなシンプルなモデルに対しては偏微分は簡単に行えるが、複雑なモデルの場合勾配の計算式を求めることが非常に困難になってしまう。
そこで、tensorflowの自動微分を使えばコードをシンプルに表すことができます。

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

n_epochs = 1000
learning_rate = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="Y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = tf.gradients(mse, [theta])[0]
training_op = tf.assign(theta, theta-learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch%100 == 0:
            print("Epoch", epoch, " MSE=", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()
    print(best_theta)
```

gradients = tf.gradients(mse, [theta])[0]はthetaについてのMSEの勾配ベクトルを計算しています。

## オプティマイザを利用

Tensorflowは勾配降下オプティマイザなどさまざまな最適化手法を提供しています。
オプティマイザを利用すれば、より簡潔なコードにすることができます。

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

n_epochs = 1000
learning_rate = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="Y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch%100 == 0:
            print("Epoch", epoch, " MSE=", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()
    print(best_theta)
```

# プレースホルダについて
プレースホルダノードは、実際には計算を行わず、実行時に出力せよと指示したデータを出力するノードです。
訓練中にTensorflowに訓練データを渡すために使います。

```python
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5

with tf.Session() as tf:
  B_val_1 = B.eval(feed_dict={A:[[1, 2, 3]]})
  B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
print(B_val_1)
print(B_val_2)
```

ミニバッチ勾配降下法を利用した線形回帰モデルの学習を通して、プレースホルダの使い方を理解しましょう。

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

n_epochs = 10
learning_rate = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

batch_size = 100
n_batches = int(np.ceil(m/batch_size))

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index) 
    indices = np.random.randint(m, size=batch_size)  
    X_batch = scaled_housing_data_plus_bias[indices] 
    y_batch = housing.target.reshape(-1, 1)[indices] 
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()
print(best_theta)
```

# モデルの保存と復元

## 保存
構築フェーズの最後にSaverノードを作ります。

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

n_epochs = 10
learning_rate = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="Y")
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(init)

  for epoch in range(n_epochs):
    if epoch%100 == 0:
     print("Epoch", epoch, "MSE=", mse.eval())
     save_path = saver.save(sess, "/tmp/my_model.ckpt")
    sess.run(training_op)

  best_theta = theta.eval()
  save_path = saver.save(sess, "/tmp/my_model_final.ckpt")

print(best_theta)
```
Saverはデフォルトではすべての変数をその名前のもとで保存、復元するが、細かく指定したい場合には、どの変数を保存して復元するか、どの名前を使うかを指定することができます。
以下のSaverは、weightという名前のもとにtheta変数だけを保存、復元します。

```python
saver = tf.train.Saver({"widths": theta})
```

## 復元
save()メソットはグラフ構造を.metaファイルに保存します。
このグラフ構造はtf.train.import_meta_graph()を使って読み込むことができます。

```python
import tensorflow as tf

saver = tf.train.import_meta_graph("/tmp/my_model_final.ckpt.meta")
theta = tf.get_default_graph().get_tensor_by_name("theta:0")

with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")
    best_theta_restored = theta.eval()

print(best_theta_restored)
```

# TensorBoardを使ったグラフと訓練曲線の可視化

ログディレクトリを作成するために、以下のコードをプログラムの冒頭に追加します。

```python
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
```

構築フェーズの末尾に以下のコードを加えます。

```python
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
```
第１行は、グラフ内にMSEを評価してsummaryというログのバイナリ列に書き込みます。
第２行は、ログディレクトリのログファイルにsummaryを書き込むために使うFileWriterを作ります。

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

learning_rate = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m/batch_size))

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index) 
    indices = np.random.randint(m, size=batch_size)  
    X_batch = scaled_housing_data_plus_bias[indices] 
    y_batch = housing.target.reshape(-1, 1)[indices] 
    return X_batch, y_batch

with tf.Session() as sess:                                                        
    sess.run(init)                                                              

    for epoch in range(n_epochs):                                              
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

file_writer.close()

print(best_theta)
```

tensorboardサーバーを立ち上げます。

```bash
tensorboard --logdir tf_logs/
```

ウェブブラウザを開き、```http://0.0.0.0:6066```に行くとTensorBoardを見ることができます。


# 名前スコープ
複雑なモデルになると、グラフ構造がわかりづらくなってしまいます。
名前スコープを利用することで、関連するノードをまとめることができます。

```python
with tf.name_scope("loss") as scope:
  error = y-pred - y
  mse = tf.reduce_mean(tf.square(error), name="mse")
```
名前スコープ内で定義された名前には、"loss/"が付けられます。

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

learning_rate = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")

with name_scape("loss") as scope:
  error = y_pred - y
  mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m/batch_size))

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index) 
    indices = np.random.randint(m, size=batch_size)  
    X_batch = scaled_housing_data_plus_bias[indices] 
    y_batch = housing.target.reshape(-1, 1)[indices] 
    return X_batch, y_batch

with tf.Session() as sess:                                                        
    sess.run(init)                                                              

    for epoch in range(n_epochs):                                              
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

file_writer.close()

print(best_theta)
```

# モジュール性
Tensorflowは、ノードを作るときにその名前がすでに存在するかどうかをチェックし、ある場合には"アンダースコア"と"インデックス"を追加して名前を一意にします。

```python
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
```

# 変数の共有
グラフの様々な構成要素の間で変数を共有したいときがあります。
共有変数がまだなければ作り、そうでなければ既存の変数を再利用するget_variable()関数を利用します。

```python
import tensorflow as tf

n_features = 3

def relu(X):
    with tf.variable_scope("relu", reuse=True):
        threshold = tf.get_variable("threshold")
        w_shape = int(X.get_shape()[1]), 1
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, threshold, name="relu")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))

relus = [relu(X) for relu_index in range(5)]
output = tf.add_n(relus, name="output")

file_writer = tf.summary.FileWriter("logs/relu6", tf.get_default_graph())
file_writer.close()
```

最初にrelu()関数を定義してから、relu/threshold変数を作り、relu()関数を呼び出して5つのReluを構築します。
relu()関数は、relu/threshold変数を再利用して他のReLUノードを作成します。



