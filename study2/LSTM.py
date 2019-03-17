import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()

# Hyper Parameters
learning_rate = 0.01    # 学习率
n_steps = 28            # LSTM 展开步数（时序持续长度）
n_inputs = 28           # 输入节点数
n_hiddens = 64         # 隐层节点数
n_layers = 2            # LSTM layer 层数
n_classes = 10          # 输出节点数（分类数目）

import _pickle as pickle
# 反序列化
with open('e:/tf/mnist/data.pkl', 'rb') as f:
    mnist = pickle.load(f)
# data
# mnist = input_data.read_data_sets("E:/Anaconda3/workspace/MNIST_data/", one_hot=True)
test_x = mnist.test.images
test_y = mnist.test.labels

# tensor placeholder
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, n_steps * n_inputs], name='x_input')     # 输入
    y = tf.placeholder(tf.float32, [None, n_classes], name='y_input')               # 输出
    keep_prob = tf.placeholder(tf.float32, name='keep_prob_input')           # 保持多少不被 dropout
    batch_size = tf.placeholder(tf.int32, [], name='batch_size_input')       # 批大小
print(batch_size.name)

# weights and biases
with tf.name_scope('weights'):
    Weights = tf.Variable(tf.truncated_normal([n_hiddens, n_classes],stddev=0.1), dtype=tf.float32, name='W')
    tf.summary.histogram('output_layer_weights', Weights)
with tf.name_scope('biases'):
    biases = tf.Variable(tf.random_normal([n_classes]), name='b')
    tf.summary.histogram('output_layer_biases', biases)

# RNN structure
def RNN_LSTM(x, Weights, biases):
    # RNN 输入 reshape
    x = tf.reshape(x, [-1, n_steps, n_inputs])
    # 定义 LSTM cell
    # cell 中的 dropout
    def attn_cell():
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hiddens)
        with tf.name_scope('lstm_dropout'):
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    # attn_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    # 实现多层 LSTM
    # [attn_cell() for _ in range(n_layers)]
    enc_cells = []
    for i in range(0, n_layers):
        enc_cells.append(attn_cell())
    with tf.name_scope('lstm_cells_layers'):
        mlstm_cell = tf.contrib.rnn.MultiRNNCell(enc_cells, state_is_tuple=True)
    # 全零初始化 state
    _init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
    # dynamic_rnn 运行网络
    outputs, states = tf.nn.dynamic_rnn(mlstm_cell, x, initial_state=_init_state, dtype=tf.float32, time_major=False)
    # 输出
    #return tf.matmul(outputs[:,-1,:], Weights) + biases
    return tf.nn.softmax(tf.matmul(outputs[:,-1,:], Weights) + biases)

with tf.name_scope('output_layer'):
    pred = RNN_LSTM(x, Weights, biases)
    tf.summary.histogram('outputs', pred)
# cost
with tf.name_scope('loss'):
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred),reduction_indices=[1]))
    tf.summary.scalar('loss', cost)
# optimizer
with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuarcy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
with tf.name_scope('accuracy'):
    accuracy = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=tf.argmax(pred, axis=1))[1]
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter("E://logs//train",sess.graph)
    test_writer = tf.summary.FileWriter("E://logs//test",sess.graph)
    # training
    step = 1
    for i in range(2000):
        _batch_size = 128
        batch_x, batch_y = mnist.train.next_batch(_batch_size)

        sess.run(train_op, feed_dict={x:batch_x, y:batch_y, keep_prob:0.5, batch_size:_batch_size})
        if (i + 1) % 100 == 0:
            #loss = sess.run(cost, feed_dict={x:batch_x, y:batch_y, keep_prob:1.0, batch_size:_batch_size})
            #acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y, keep_prob:1.0, batch_size:_batch_size})
            #print('Iter: %d' % ((i+1) * _batch_size), '| train loss: %.6f' % loss, '| train accuracy: %.6f' % acc)
            train_result = sess.run(merged, feed_dict={x:batch_x, y:batch_y, keep_prob:1.0, batch_size:_batch_size})
            test_result = sess.run(merged, feed_dict={x:test_x, y:test_y, keep_prob:1.0, batch_size:test_x.shape[0]})
            train_writer.add_summary(train_result,i+1)
            test_writer.add_summary(test_result,i+1)

    print("Optimization Finished!")
    # prediction
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x:test_x, y:test_y, keep_prob:1.0, batch_size:test_x.shape[0]}))


PS E:\logs\test> tensorboard.exe --logdir=E:\logs\test
TensorBoard 0.4.0 at http://KOTIN:6006 (Press CTRL+C to quit)


###############################################################
with tf.name_scope("my_name_scope"):
    v2 = tf.Variable(1, name="var2", dtype=tf.float32)
print(v2.name)

###############################################################
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28   # MNIST data input (img shape: 28*28)
n_steps = 28    # time steps
n_hidden_units = 128   # neurons in hidden layer
n_classes = 10      # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

a=weights['in']
b=tf.Session()
b.run([a])

def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)\

    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
            }))
        step += 1


#############################################################
import logging

class BasicLSTMCell(RNNCell):
    """Basic LSTM recurrent network cell.

      The implementation is based on: http://arxiv.org/abs/1409.2329.

      We add forget_bias (default: 1) to the biases of the forget gate in order to
      reduce the scale of forgetting in the beginning of the training.

      It does not allow cell clipping, a projection layer, and does not
      use peep-hole connections: it is the basic baseline.

      For advanced models, please use the full LSTMCell that follows."""

    def __init__(self, num_units, forget_bias=1.0, input_size=None,
                   state_is_tuple=True, activation='tanh'):
        """Initialize the basic LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """
        if not state_is_tuple:
            logging.warning("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
             logging.warning("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation


    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with vs.variable_scope(scope or "basic_lstm_cell"):
      # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

# 　　　 # 线性计算 concat = [inputs, h]W + b
# 　　　 # 线性计算，分配W和b，W的shape为（2*num_units, 4*num_units）, b的shape为（4*num_units,）,共包含有四套参数，
#       # concat shape(batch_size, 4*num_units)
#    　　# 注意：只有cell 的input和output的size相等时才可以这样计算，否则要定义两套W,b.每套再包含四套参数
        concat = _linear([inputs, h], 4 * self._num_units, True, scope=scope)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

      new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * sigmoid(o)

      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c, new_h)
      else:
        new_state = array_ops.concat([new_c, new_h], 1)
      return new_h, new_state

help(LSTMStateTuple)


print(tf.contrib.rnn.LSTMStateTuple)
tf.contrib.rnn.LSTMStateTuple(1,1)


tf.python.ops.array_ops

biases=tf.Variable(tf.zeros([2,3]))#定义一个2x3的全0矩阵
sess=tf.InteractiveSession()#使用InteractiveSession函数
biases.initializer.run()#使用初始化器 initializer op 的 run() 方法初始化 'biases'
print(sess.run(biases))#输出变量值

import tensorflow as tf

v1 = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="v1")
v2 = tf.Variable(tf.zeros([200]), name="v2")
saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    saver.save(sess, "checkpoint/model_test", global_step=1)

v1.initializer.run()#使用初始化器 initializer op 的 run() 方法初始化 'biases'
sess.run(tf.global_variables_initializer())
sess.run(v1.initializer)
print(sess.run(v1))

import tensorflow as tf

v1 = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="v1")
v2 = tf.Variable(tf.zeros([200]), name="v2")
saver = tf.train.Saver()
with tf.Session() as sess:
    # init_op = tf.global_variables_initializer()
    # sess.run(init_op)
    saver.restore(sess, "checkpoint/model_test-1")
    # saver.save(sess,"checkpoint/model_test",global_step=1)

#####################################################
import numpy as np
# 保存神经网络参数
def save_para():
    # 定义权重参数
    W = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype = tf.float32, name = 'weights')
    # 定义偏置参数
    b = tf.Variable([[1, 2, 3]], dtype = tf.float32, name = 'biases')
    # 参数初始化
    init = tf.global_variables_initializer()
    # 定义保存参数的saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        # 保存session中的数据
        save_path = saver.save(sess, 'my_net/save_net.ckpt')
        # 输出保存路径
        print('Save to path: ', save_path)
import os
os.getcwd()
# 恢复神经网络参数
def restore_para():
    # 定义权重参数
    W = tf.Variable(np.arange(6).reshape((2, 3)), dtype = tf.float32, name = 'weights')
    # 定义偏置参数
    b = tf.Variable(np.arange(3).reshape((1, 3)), dtype = tf.float32, name = 'biases')
    # 定义提取参数的saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 加载文件中的参数数据，会根据name加载数据并保存到变量W和b中
        save_path = saver.restore(sess, 'my_net/save_net.ckpt')
        # 输出保存路径
        print('Weights: ', sess.run(W))
        print('biases:  ', sess.run(b))


# save_para()
restore_para()
