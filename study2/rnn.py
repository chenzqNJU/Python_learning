
import tensorflow as tf
import numpy as np


n_inputs = 3
# hidden state
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

# 由于Wx要和X相乘，故低维是n_inputs
Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons],dtype=tf.float32))
# 低维，高维都是n_neurons，为了使得输出也是hidden state的深度
# 这样下一次才可以继续运算
Wy = tf.Variable(tf.random_normal(shape=[n_neurons,n_neurons],dtype=tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))

# Y0初始化为0，初始时没有记忆
Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
# 把上一轮输出Y0也作为输入
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)
init = tf.global_variables_initializer()
X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1
with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})

# Y0，Y1都是4*5大小，4是mini-batch数目，5是输出神经元个数

# 这种和上面那种手动实现的效果相同
n_inputs = 3
n_neurons = 5
X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)
Y0, Y1 = output_seqs
# run部分
init = tf.global_variables_initializer()
X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]])

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})


###############################################################
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10
learning_rate = 0.001

tf.reset_default_graph()
sess.close()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# 一维输出
y = tf.placeholder(tf.int32, [None])
# 使用最简单的basicRNNcell
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
#使用dynamic_rnn
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
# 原始输出
logits = fully_connected(states, n_outputs, activation_fn=None)
# 计算和真实的交叉熵
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
# 使用AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
# 计算准确率，只有等于y才是对的，其他都错
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()


from tensorflow.examples.tutorials.mnist import input_data
input_data.read_data_sets("/home/wd/MNIST_data",one_hot=True)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")


img0 = mnist.train.images[0].reshape(28,28)

# 转换到合理的输入shape
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels
# run100遍，每次处理150个输入
n_epochs = 100
batch_size = 150
# 开始循环
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            # 读入数据并reshape
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            # X大写，y小写
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        # 每次打印一下当前信息
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)


sess=tf.Session()
init.run()
init = tf.global_variables_initializer()
epoch=1;iteration=1
for epoch in range(n_epochs):


    for iteration in range(mnist.train.num_examples // batch_size):

        # 读入数据并reshape
        X_batch, y_batch = mnist.train.next_batch(batch_size)
        X_batch = X_batch.reshape((-1, n_steps, n_inputs))
        # X大写，y小写
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    # 每次打印一下当前信息
    print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)


y_batch.shape

x = tf.placeholder(tf.int32, shape=[None])
print(x.get_shape())
# ==> '(4,)'


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

input1 = tf.placeholder(tf.float32,[None,2])
input2 = tf.placeholder(tf.float32,[None,2])
output = tf.multiply(input1, input2)
sess=tf.Session()
sess.run([output], feed_dict={input1:[[7.,2],[1,2]], input2:[[2.,3],[1,2]]})
print(input2.get_shape())
y_batch.shape

sess.close()

n_inputs = 2
n_steps = 5

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
n_neurons = 100
n_layers = 3 # 做了3层rnn
# 模型不是越复杂越好，越复杂所需数据量越大，否则会有过拟合的风险
# 可以加dropout来控制
layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
          for layer in range(n_layers)]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
init = tf.global_variables_initializer()
X_batch = np.random.rand(2, n_steps, n_inputs)
with tf.Session() as sess:
    init.run()
    outputs_val, states_val = sess.run([outputs, states], feed_dict={X: X_batch})


outputs_val.shape

######################################################## TensorFlow中LSTM具体实现
n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10
n_layers = 3

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
              for layer in range(n_layers)]
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
top_layer_h_state = states[-1][1]
logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()


##########################################################
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

sess.run(W)
W.get_shape()







