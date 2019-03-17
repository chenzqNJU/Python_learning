import numpy as np


import pandas as pd

import re

import tensorflow as tf
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
c = a * b
# 创建会话
sess = tf.Session()
# 计算 c
print(sess.run(c)) # 进行矩阵乘法，输出[3., 8.]
sess.close()

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tflearn.datasets.oxflower17 as oxflower17
X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))
Y.shape

X[1,1,1,:]

network = input_data(shape=[None, 227, 227, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 17, activation='softmax')
network = regression(network, optimizer='momentum',loss='categorical_crossentropy',learning_rate=0.001) # 回归操作，同时规定网络所使用的学习率、损失函数和优化器
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,show_metric=True,
          batch_size=64, snapshot_step=200,snapshot_epoch=False, run_id='alexnet_oxflowers17')


from sklearn import preprocessing

enc = preprocessing.OneHotEncoder(categories='auto')  # 创建对象
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  # 拟合
array = enc.transform([[0, 1, 3]]).toarray()  # 转化
print(array)

enc.transform([[0, 0, 0]]).toarray()

X[1,2]



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

tag_list = ['青年 吃货 唱歌',
            '少年 游戏 叛逆',
            '少年 吃货 足球']

vectorizer = CountVectorizer() #将文本中的词语转换为词频矩阵


X = vectorizer.fit_transform(tag_list) #计算个词语出现的次数
X.toarray()
"""
word_dict = vectorizer.vocabulary_
{'唱歌': 2, '吃货': 1, '青年': 6, '足球': 5, '叛逆': 0, '少年': 3, '游戏': 4}
"""

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)  #将词频矩阵X统计成TF-IDF值
print(tfidf.toarray())

import tensorflow as tf
import numpy as np
x_data = np.linspace(-1,1,300)[:, np.newaxis] # 为了使点更密一些，我们构建了 300 个点，分


noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

inputs=x_data;in_size=1;out_size=20
def add_layer(inputs, in_size, out_size, activation_function=None):
    # 构建权重： in_size×out_size 大小的矩阵
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 构建偏置： 1×out_size 的矩阵
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 矩阵相乘
    Wx_plus_b = tf.matmul(tf.cast(inputs,tf.float32), weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs # 得到输出数据
# 构建隐藏层，假设隐藏层有 10 个神经元
h1 = add_layer(x_data, 1, 20, activation_function=tf.nn.relu)
# 构建输出层，假设输出层和输入层一样，有 1 个神经元
prediction = add_layer(h1, 20, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer() # 初始化所有变量
sess = tf.Session()
sess.run(init)

for i in range(1000): # 训练 1000 次
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0: # 每 50 次打印出一次损失值
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



###############################################
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])

init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))



optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))

##########################################
import tensorflow as tf
import numpy as np

# Declare list of features. We only have one numeric feature. There are many
# other types of columns that are more complicated and useful.
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# linear classification, and many neural network classifiers and regressors.
# The following code provides an estimator that does linear regression.
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use two data sets: one for training and one for evaluation
# We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.train(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did.
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r" % train_metrics)
print("eval metrics: %r" % eval_metrics)