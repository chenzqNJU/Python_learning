import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint
import myfunc
from importlib import reload

reload(myfunc)



## 目录下所有文件
def all_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result


test = r'G:\stk'
result = all_path(test)

stk = [each for each in result if '002795' in each]

len_day = len(stk)
pprint.pprint(stk)

df1 = pd.DataFrame()
for a in stk:
    try:
        df = pd.read_csv(a)
    except:
        df = pd.read_csv(a, encoding='gbk')
    df['date'] = a.split('\\')[2]
    df1 = df1.append(df)
#观测11:30的
# a=df1[df1.time.str.startswith('11:30')]

df2 = df1.reset_index().drop(['index', 'Unnamed: 0', 'change'], axis=1)
df2.code = df2.code.astype(str).str.zfill(6)
temp = dict({"买盘": 0, '卖盘': 1, '中性盘': 2})
df2.type = df2.type.apply(lambda x: temp[x]).astype('int8')
df2['datetime'] = pd.to_datetime(df2.date + ' ' + df2.time)
df2.drop(['time', 'date'], axis=1, inplace=True)

df2.datetime = df2.datetime.apply(lambda x: x.replace(second=0))

df3 = df2.groupby('datetime')['volume', 'amount'].sum()
df4 = df2.groupby('datetime')['price'].last()
df = df3.join(df4)
del df3;
del df4

############################ 5 分钟
from datetime import timedelta
import datetime

df.reset_index(inplace=True)
# df['d'] = df.datetime + timedelta(hours=1)
df['min'] = np.mod(df.datetime.dt.minute, 5) * timedelta(minutes=1)
###### 将3:00的归到2:55-2:59 11:30的归到
tmp1 = df.datetime.dt.hour == 15
tmp2 = (df.datetime.dt.hour == 11) & (df.datetime.dt.minute == 30)

df.loc[(tmp1 | tmp2), 'min'] = 5 * timedelta(minutes=1)

df.datetime = df.datetime - df['min']
del df['min']
df3 = df.groupby('datetime')['volume', 'amount'].sum()
df4 = df.groupby('datetime')['price'].last()
df = df3.join(df4)
del df3;
del df4

length = len(df)
train = df.reset_index()

##################################
# def make_train_test_csv(cls, orgin_data_path=None, all_data_path=None, time_step=60):

# 读取原始数据，只保留需要使用的列
total_data = df.reset_index()
# 根据股票代码排序，相同的股票代码按照交易日期排序。
# inplace参数表示不需要返回排序后的结果，直接覆盖原变量即可
total_data.sort_values(by='datetime', inplace=True)

# 根据股票day分组
g_stock_num = total_data.groupby(by=total_data.datetime.dt.date)
# gate = g_stock_num["price"].apply(lambda x: x[:1]).pct_change().dropna()
# help(pd.DataFrame.pct_change)
# gate = gate.reset_index(level=1, drop=True)
# gate.index = gate.index.values - timedelta(days=1)
gate = g_stock_num["price"].apply(lambda x: x[:1]).pct_change().shift(-1)
gate = gate.reset_index(level=1, drop=True)
gate.name='rate_cl'

# help(pd.DataFrame.reset_index)
# gate.index.levels[0]
# gate.index.labels[1]
# gate.index.names
# list(zip(*gate.index.values))

# 开盘涨跌幅
# gate1 = g_stock_num["price"].apply(lambda x: x.iloc[[0, -1]]).reset_index()
# a=gate1.groupby('datetime').apply(lambda x:range(len(x)))
# # b = a.values.tolist()
# from itertools import chain
# gate1['flag'] = list(chain(*a))
# gate1 = gate1.pivot(index='datetime',columns='flag',values='price')


g_stock_num = total_data.groupby(by=total_data.datetime.dt.date,as_index=True)
t1 = g_stock_num[["price"]].first()
t2 = g_stock_num[["price"]].last()
t2_=t2.shift(-1).rename(columns={'price':'price_op'})
t= t1.join(t2_)
gate1=(t.price_op - t.price) / t.price
gate1.name='rate_op'
# g_stock_num["price"].agg([np.sum,np.mean])
# g_stock_num["price"].first()
# g_stock_num["price"].last()
# g_stock_num["price"].tail()
# g_stock_num["price"].head()
# g_stock_num["price"].sum()


# 重新调整列的顺序，为接下来处理成输入、输出形式做准备
total_data['date'] = total_data.datetime.dt.date
total_data['time'] = total_data.datetime.dt.time
# 转成多列
t = total_data.pivot(index='date',columns='time',values=['volume','price'])
t.columns=[x + str(y) for x,y in t.columns.values]
# 可能某个时刻没有成交量，为none
t.fillna(0,inplace=True)

total_data=t.join([gate,gate1])

# 拿time_step个交易日的数据（默认为60个交易日），进行标准化
n_dim = 98
time_step=20
data_one_stock_num=total_data
def func_stand(total_data, time_step):
    # 通过apply进入函数内的数据，其股票名为data_one_stock_num.name，类型为pd.dataFrame
    # 即，进入此函数的数据为所有名为data_one_stock_num.name的集合
    # dataFrame.shape:(num , 11), num是这个股票出现的次数
    data_one_stock_num = total_data.copy()
    for colu_name in data_one_stock_num.columns:
        if colu_name.startswith('rate'):
            continue
        data_one_stock_num[colu_name] = (
                (data_one_stock_num[colu_name] - data_one_stock_num[colu_name].rolling(time_step).mean()) /
                data_one_stock_num[colu_name].rolling(time_step).std())
    return data_one_stock_num
data_one_stock_num = func_stand(total_data,20)
data_one_stock_num.dropna(inplace=True)
data_one_stock_num=data_one_stock_num.astype('float16')
#这里删了20个（前面19个 最后1个）

# 将经过标准化的数据处理成训练集和测试集可接受的形式
def func_train_test_data(data_one_stock_num, time_step):
    # 提取输入列（对应train_x）
    data_temp_x = data_one_stock_num.iloc[:,:-2]
    # 提取输出列（对应train_y）
    data_temp_y = data_one_stock_num.iloc[:,-2:]
    data_res = []
    # a=data_temp_x.iloc[i - time_step + 1: i + 1].values.reshape(1, time_step * n_dim).tolist()
    # b=a.astype('float16')
    # i=time_step - 1
    for i in range(time_step - 1, len(data_temp_x.index)):
        data_res.append(data_temp_x.iloc[i - time_step + 1: i + 1].values.reshape(1, time_step * n_dim).tolist() +
                        data_temp_y.iloc[i][['rate_cl','rate_op']].values.reshape(1, 2).tolist())
    # if len(data_res) != 0:
    #       pd.DataFrame(data_res).to_csv(all_data_path, index=False, header=False, mode="a")
    return pd.DataFrame(data_res)


data_res = func_train_test_data(data_one_stock_num,time_step=time_step)

import tensorflow as tf
from tensorflow.contrib import rnn


class Deeplearing():
    def stock_lstm(self, ):

        basic_path = r'E:\tf\dlstock-master'

        # 定义存储模型的文件路径
        # os.path.join(basic_path, "stock_rnn_save.ckpt")表示在运行的python文件路径下保存，文件名为stock_rnn.ckpt，在我环境下运行总是提示“另一个程序正在使用此文件，进程无法访问”，换到其他路径就OK
        model_save_path = "e:\\tf\\save\\stock.ckpt"  # os.path.join(basic_path, "stock_rnn_save.ckpt")
        # 定义训练集的文件路径，当前为运行的python文件路径下，文件名为train_data.csv
        train_csv_path = os.path.join(basic_path, "train_data.csv")
        # 定义测试集的文件路径，当前为运行的python文件路径下，文件名为test_data.csv
        test_csv_path = os.path.join(basic_path, "test_data.csv")
        # 学习率
        learning_rate = 0.001
        origin_data_row = 60
        origin_data_col = 7
        layer_num = 2
        cell_num = 256
        output_num = 1
        batch_size = tf.placeholder(tf.int32, [])

        W = {
            'in': tf.Variable(tf.truncated_normal([origin_data_col, cell_num], stddev=1), dtype=tf.float32),
            'out': tf.Variable(tf.truncated_normal([cell_num, output_num], stddev=1), dtype=tf.float32)
        }
        bias = {
            'in': tf.Variable(tf.constant(0.1, shape=[cell_num, ]), dtype=tf.float32),
            'out': tf.Variable(tf.constant(0.1, shape=[output_num, ]), dtype=tf.float32)
        }
        input_x = tf.placeholder(tf.float32, [None, origin_data_col * origin_data_row])
        input_y = tf.placeholder(tf.float32, [None, output_num])
        keep_prob = tf.placeholder(tf.float32, [])
        input_x_after_reshape_2 = tf.reshape(input_x, [-1, origin_data_col])

        input_rnn = tf.nn.dropout(tf.nn.relu_layer(input_x_after_reshape_2, W['in'], bias['in']), keep_prob)

        input_rnn = tf.reshape(input_rnn, [-1, origin_data_row, cell_num])

        # 定义一个带着“开关”的LSTM单层，一般管它叫细胞
        def lstm_cell():
            cell = rnn.LSTMCell(cell_num, reuse=tf.get_variable_scope().reuse)
            return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

        lstm_layers = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
        # 初始化LSTM网络
        init_state = lstm_layers.zero_state(batch_size, dtype=tf.float32)

        outputs, state = tf.nn.dynamic_rnn(lstm_layers, inputs=input_rnn, initial_state=init_state, time_major=False)
        h_state = state[-1][1]
        y_pre = tf.matmul(h_state, W['out']) + bias['out']
        loss = tf.reduce_mean(tf.square(tf.subtract(y_pre, input_y)))
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        accuracy = tf.reduce_max(tf.abs(tf.subtract(y_pre, input_y)))

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        data_get = get_stock_data()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            try:
                saver.restore(sess, model_save_path)
                print("成功加载模型参数")
            except:
                print("未加载模型参数，文件被删除或者第一次运行")
                sess.run(init)

            # 给batch_size赋值
            _batch_size = 200
            for i in range(20000):
                try:
                    # 读取训练集数据
                    train_x, train_y = data_get.get_train_test_data_new(batch_size=_batch_size,
                                                                        file_path=train_csv_path)
                except StopIteration:
                    print("训练集均已训练完毕")
                    train_accuracy = sess.run(accuracy, feed_dict={
                        input_x: train_x, input_y: train_y, keep_prob: 1.0, batch_size: _batch_size})
                    print("step: {0}, training_accuracy: {1}".format(i + 1, train_accuracy))
                    saver.save(sess, model_save_path)
                    print("保存模型\n")
                    break

                if (i + 1) % 20 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={
                        input_x: train_x, input_y: train_y, keep_prob: 1.0, batch_size: _batch_size})
                    # 输出
                    print("step: {0}, training_accuracy: {1}".format(i + 1, train_accuracy))
                    saver.save(sess, model_save_path)
                    print("保存模型\n")
                    ############################################
                    # 这部分代码作用为：每次保存模型，顺便将预测收益和真实收益输出保存至show_y_pre.txt文件下。熟悉tf可视化，完全可以采用可视化替代
                    _y_pre_train = sess.run(y_pre, feed_dict={
                        input_x: train_x, input_y: train_y, keep_prob: 1.0, batch_size: _batch_size})
                    _loss = sess.run(loss, feed_dict={
                        input_x: train_x, input_y: train_y, keep_prob: 1.0, batch_size: _batch_size})
                    a1 = np.array(train_y).reshape(1, _batch_size)
                    b1 = np.array(_y_pre_train).reshape(1, _batch_size)
                    with open(os.path.join(basic_path, "show_y_pre.txt"), "w") as f:
                        f.write(str(a1.tolist()))
                        f.write("\n")
                        f.write(str(b1.tolist()))
                        f.write("\n")
                        f.write(str(_loss))
                    ############################################
                # 按照给定的参数训练一次LSTM神经网络
                sess.run(train_op,
                         feed_dict={input_x: train_x, input_y: train_y, keep_prob: 0.6, batch_size: _batch_size})

            # 计算测试数据的准确率
            # 读取测试集数据
            test_size = 100
            test_x, test_y = data_get.get_train_test_data_new(batch_size=test_size, file_path=test_csv_path)
            print("test accuracy {0}".format(sess.run(accuracy, feed_dict={
                input_x: test_x, input_y: test_y, keep_prob: 1.0, batch_size: test_size})))

    # def stock_lstm_softmax(self, ):
    #     """
    #     使用LSTM处理股票数据
    #     分类预测
    #     """
    #     # 获取当前python文件运行在哪个目录下并去掉最后的文件名，如：F:/deeplearning/main.py --> F:/deeplearning
    #     # 在linux下同样起作用
    #     basic_path = os.path.dirname(os.path.abspath(__file__))
    #     # 定义存储模型的文件路径，当前为运行的python文件路径下，文件名为stock_rnn.ckpt
    #     model_save_path = r"F:\\123\\save\\stock_rnn_save.ckpt"  # os.path.join(basic_path, "stock_rnn.ckpt")
    #     # 定义训练集的文件路径，当前为运行的python文件路径下，文件名为train_data.csv
    #     train_csv_path = os.path.join(basic_path, "train_data.csv")
    #     # 定义测试集的文件路径，当前为运行的python文件路径下，文件名为test_data.csv
    #     test_csv_path = os.path.join(basic_path, "test_data.csv")
    #     # 学习率
    #     learning_rate = 0.001
    #     # 喂数据给LSTM的原始数据有几行，即：一次希望LSTM能“看到”多少个交易日的数据
    #     origin_data_row = 60
    #     # 喂给LSTM的原始数据有几列，即：日线数据有几个元素
    #     origin_data_col = 7
    #     # LSTM网络有几层
    #     layer_num = 20
    #     # LSTM网络，每层有几个神经元
    #     cell_num = 256
    #     # 最后输出的数据维度，即：要预测几个数据，该处需要处理分类问题，按照自己设定的类型数量设定
    #     output_num = 3
    #     # 每次给LSTM网络喂多少行经过处理的股票数据。该参数依据自己显卡和网络大小动态调整，越大 一次处理的就越多，越能占用更多的计算资源
    #     batch_size = tf.placeholder(tf.int32, [])
    #     # 输入层、输出层权重、偏置。
    #     # 通过这两对参数，LSTM层能够匹配输入和输出的数据
    #     W = {
    #         'in': tf.Variable(tf.truncated_normal([origin_data_col, cell_num], stddev=1), dtype=tf.float32),
    #         'out': tf.Variable(tf.truncated_normal([cell_num, output_num], stddev=1), dtype=tf.float32)
    #     }
    #     bias = {
    #         'in': tf.Variable(tf.constant(0.1, shape=[cell_num, ]), dtype=tf.float32),
    #         'out': tf.Variable(tf.constant(0.1, shape=[output_num, ]), dtype=tf.float32)
    #     }
    #     # 告诉LSTM网络，即将要喂的数据是几行几列
    #     # None的意思就是喂数据时，行数不确定交给tf自动匹配
    #     # 我们喂得数据行数其实就是batch_size，但是因为None这个位置tf只接受数字变量，而batch_size是placeholder定义的Tensor变量，表示我们在喂数据的时候才会告诉tf具体的值是多少
    #     input_x = tf.placeholder(tf.float32, [None, origin_data_col * origin_data_row])
    #     input_y = tf.placeholder(tf.float32, [None, output_num])
    #     # 处理过拟合问题。该值在其起作用的层上，给该层每一个神经元添加一个“开关”，“开关”打开的概率是keep_prob定义的值，一旦开关被关了，这个神经元的输出将被“阻断”。这样做可以平衡各个神经元起作用的重要性，杜绝某一个神经元“一家独大”，各种大佬都证明这种方法可以有效减弱过拟合的风险。
    #     keep_prob = tf.placeholder(tf.float32, [])
    #
    #     # 通过reshape将输入的input_x转化成2维，-1表示函数自己判断该是多少行，列必须是origin_data_col
    #     # 转化成2维 是因为即将要做矩阵乘法，矩阵一般都是2维的（反正我没见过3维的）
    #     input_x_after_reshape_2 = tf.reshape(input_x, [-1, origin_data_col])
    #
    #     # 当前计算的这一行，就是输入层。输入层的激活函数是relu,并且施加一个“开关”，其打开的概率为keep_prob
    #     # input_rnn即是输入层的输出，也是下一层--LSTM层的输入
    #     input_rnn = tf.nn.dropout(tf.nn.relu_layer(input_x_after_reshape_2, W['in'], bias['in']), keep_prob)
    #
    #     # 通过reshape将输入的input_rnn转化成3维
    #     # 转化成3维，是因为即将要进入LSTM层，接收3个维度的数据。粗糙点说，即LSTM接受：batch_size个，origin_data_row行cell_num列的矩阵，这里写-1的原因与input_x写None一致
    #     input_rnn = tf.reshape(input_rnn, [-1, origin_data_row, cell_num])
    #
    #     # 定义一个带着“开关”的LSTM单层，一般管它叫细胞
    #     def lstm_cell():
    #         cell = rnn.LSTMCell(cell_num, reuse=tf.get_variable_scope().reuse)
    #         return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    #
    #     # 这一行就是tensorflow定义多层LSTM网络的代码
    #     lstm_layers = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
    #     # 初始化LSTM网络
    #     init_state = lstm_layers.zero_state(batch_size, dtype=tf.float32)
    #
    #     # 使用dynamic_rnn函数，告知tf构建多层LSTM网络，并定义该层的输出
    #     outputs, state = tf.nn.dynamic_rnn(lstm_layers, inputs=input_rnn, initial_state=init_state, time_major=False)
    #     h_state = state[-1][1]
    #
    #     # 该行代码表示了输出层
    #     # 将LSTM层的输出，输入到输出层（输出层带softmax激活函数），输出为各个分类的概率
    #     # 假设有3个分类，那么输出举例为：[0.001, 0.992, 0.007]，表示第1种分类概率千分之1，第二种99.2%, 第三种千分之7
    #     y_pre = tf.nn.softmax(tf.matmul(h_state, W['out']) + bias['out'])
    #
    #     # 损失函数，用作指导tf
    #     # loss定义为交叉熵损失函数，softmax输出层大多都使用的这个损失函数。关于该损失函数详情可以百度下
    #     loss = -tf.reduce_mean(input_y * tf.log(y_pre))
    #     # 告诉tf，它需要做的事情就是就是尽可能将loss减小
    #     # learning_rate是减小的这个过程中的参数。如果将我们的目标比喻为“从北向南走路走到菜市场”，我理解的是
    #     # learning_rate越大，我们走的每一步就迈的越大。初看似乎步子越大越好，但是其实并不能保证每一步都是向南走
    #     # 的，有可能因为训练数据的原因，导致我们朝西走了一大步。或者我们马上就要到菜市场了，但是一大步走过去，给
    #     # 走过了。。。综上，这个learning_rate（学习率参数）的取值，无法给出一个比较普适的，还是需要根据实际情况去
    #     # 尝试和调整。0.001的取值是tf给的默认值
    #     # 上述例子是个人理解用尽可能通俗易懂地语言表达。如有错误，欢迎指正
    #     train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #
    #     # 这块定义了一个新的值，用作展示训练的效果
    #     # 它的定义为：预测对的 / 总预测数，例如：0.55表示预测正确了55%
    #     correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(input_y, 1))
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #
    #     # 用以保存参数的函数（跑完下次再跑，就可以直接读取上次跑的结果而不必从头开始）
    #     saver = tf.train.Saver(tf.global_variables())
    #
    #     # tf要求必须如此定义一个init变量，用以在初始化运行（也就是没有保存模型）时加载各个变量
    #     init = tf.global_variables_initializer()
    #     # 获取数据（这是我们自己定义的类）
    #     data_get = get_stock_data()
    #     # 设置tf按需使用GPU资源，而不是有多少就占用多少
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #
    #     # 使用with，保证执行完后正常关闭tf
    #     with tf.Session(config=config) as sess:
    #         try:
    #             # 定义了存储模型的文件路径，即：当前运行的python文件路径下，文件名为stock_rnn.ckpt
    #             saver.restore(sess, model_save_path)
    #             print("成功加载模型参数")
    #         except:
    #             # 如果是第一次运行，通过init告知tf加载并初始化变量
    #             print("未加载模型参数，文件被删除或者第一次运行")
    #             sess.run(init)
    #
    #         # 给batch_size赋值
    #         _batch_size = 200
    #         for i in range(20000):
    #             try:
    #                 # 读取训练集数据
    #                 train_x, train_y = data_get.get_train_test_data_softmax(batch_size=_batch_size,
    #                                                                         file_path=train_csv_path)
    #             except StopIteration:
    #                 print("训练集均已训练完毕")
    #                 train_accuracy = sess.run(accuracy, feed_dict={
    #                     input_x: train_x, input_y: train_y, keep_prob: 1.0, batch_size: _batch_size})
    #                 print("step: {0}, training_accuracy: {1}".format(i + 1, train_accuracy))
    #                 saver.save(sess, model_save_path)
    #                 print("保存模型\n")
    #                 break
    #
    #             if (i + 1) % 20 == 0:
    #                 train_accuracy = sess.run(accuracy, feed_dict={
    #                     input_x: train_x, input_y: train_y, keep_prob: 1.0, batch_size: _batch_size})
    #                 print("step: {0}, training_accuracy: {1}".format(i + 1, train_accuracy))
    #                 saver.save(sess, model_save_path)
    #                 print("保存模型\n")
    #             # 按照给定的参数训练一次LSTM神经网络
    #             sess.run(train_op,
    #                      feed_dict={input_x: train_x, input_y: train_y, keep_prob: 0.6, batch_size: _batch_size})
    #
    #         # 计算测试数据的准确率
    #         # 读取测试集数据
    #         test_size = 100
    #         test_x, test_y = data_get.get_train_test_data_softmax(batch_size=_batch_size, file_path=test_csv_path)
    #         print("test accuracy {0}".format(sess.run(accuracy, feed_dict={
    #             input_x: test_x, input_y: test_y, keep_prob: 1.0, batch_size: _batch_size})))


len(df)

df.iloc[0, 0] - timedelta(minutes=1) * 3
df.iloc[0, -1]
df.iloc[0, 0] + df.iloc[0, -1]

timedelta(minutes=1) * 2

np.mod(4, 2)
timedelta(hours=1) * 2
a = df.index[1]

a = df.min
b = np.mod(df.datetime.dt.minute, 5)
c = b * timedelta(minutes=1)
d = c[1]
f = timedelta(minutes=1)

nation_day = pd.Timestamp("2018-10-1")
nation_day + f
df.iloc[0, 0] + f
a = pd.Timestamp.now()
b = datetime.datetime.now()

df2.datetime.dt.second

help(pd.DataFrame.drop)

a = df1.time[1]
a = df2.datetime
b = a[1]
b.second = 2
b.replace(second=6)

len(df1)
