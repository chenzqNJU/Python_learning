def four_layer_auto_encoder():
    '''
    通过构建一个2维的自编码网络，将MNIST数据集的数据特征提取处来，并通过这些特征重建一个MNIST数据集
    这里使用4层逐渐压缩将785维度分别压缩成256,64,16,2这4个特征向量,最后再还原
    '''

    '''
    导入MNIST数据集
    '''
    # mnist是一个轻量级的类，它以numpy数组的形式存储着训练，校验，测试数据集  one_hot表示输出二值化后的10维
    mnist = input_data.read_data_sets('MNIST-data', one_hot=True)

    print(type(mnist))  # <class 'tensorflow.contrib.learn.python.learn.datasets.base.Datasets'>

    print('Training data shape:', mnist.train.images.shape)  # Training data shape: (55000, 784)
    print('Test data shape:', mnist.test.images.shape)  # Test data shape: (10000, 784)
    print('Validation data shape:', mnist.validation.images.shape)  # Validation data shape: (5000, 784)
    print('Training label shape:', mnist.train.labels.shape)  # Training label shape: (55000, 10)

    '''
    定义参数，以及网络结构
    '''
    n_input = 784  # 输入节点
    n_hidden_1 = 256
    n_hidden_2 = 64
    n_hidden_3 = 16
    n_hidden_4 = 2
    batch_size = 256  # 小批量大小
    training_epochs = 20  # 迭代轮数
    display_epoch = 5  # 迭代1轮输出5次信息
    learning_rate = 1e-2  # 学习率
    show_num = 10  # 显示的图片个数

    # 定义占位符
    input_x = tf.placeholder(dtype=tf.float32, shape=[None, n_input])  # 输入
    input_y = input_x  # 输出

    # 学习参数
    weights = {
        'encoder_h1': tf.Variable(tf.random_normal(shape=[n_input, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal(shape=[n_hidden_1, n_hidden_2])),
        'encoder_h3': tf.Variable(tf.random_normal(shape=[n_hidden_2, n_hidden_3])),
        'encoder_h4': tf.Variable(tf.random_normal(shape=[n_hidden_3, n_hidden_4])),
        'decoder_h1': tf.Variable(tf.random_normal(shape=[n_hidden_4, n_hidden_3])),
        'decoder_h2': tf.Variable(tf.random_normal(shape=[n_hidden_3, n_hidden_2])),
        'decoder_h3': tf.Variable(tf.random_normal(shape=[n_hidden_2, n_hidden_1])),
        'decoder_h4': tf.Variable(tf.random_normal(shape=[n_hidden_1, n_input]))
    }

    biases = {
        'encoder_b1': tf.Variable(tf.random_normal(shape=[n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal(shape=[n_hidden_2])),
        'encoder_b3': tf.Variable(tf.random_normal(shape=[n_hidden_3])),
        'encoder_b4': tf.Variable(tf.random_normal(shape=[n_hidden_4])),
        'decoder_b1': tf.Variable(tf.random_normal(shape=[n_hidden_3])),
        'decoder_b2': tf.Variable(tf.random_normal(shape=[n_hidden_2])),
        'decoder_b3': tf.Variable(tf.random_normal(shape=[n_hidden_1])),
        'decoder_b4': tf.Variable(tf.random_normal(shape=[n_input]))
    }

    # 编码
    encoder_h1 = tf.nn.sigmoid(tf.add(tf.matmul(input_x, weights['encoder_h1']), biases['encoder_b1']))
    encoder_h2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_h1, weights['encoder_h2']), biases['encoder_b2']))
    encoder_h3 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_h2, weights['encoder_h3']), biases['encoder_b3']))
    # 在编码的最后一层，没有进行sigmoid变化，这是因为生成的二维特征数据其特征已经标的极为主要，所有我们希望让它
    # 传到解码器中，少一些变化可以最大化地保存原有的主要特征
    # encoder_h4 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_h3,weights['encoder_h4']),biases['encoder_b4']))
    encoder_h4 = tf.add(tf.matmul(encoder_h3, weights['encoder_h4']), biases['encoder_b4'])

    # 解码
    decoder_h1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_h4, weights['decoder_h1']), biases['decoder_b1']))
    decoder_h2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_h1, weights['decoder_h2']), biases['decoder_b2']))
    decoder_h3 = tf.nn.sigmoid(tf.add(tf.matmul(decoder_h2, weights['decoder_h3']), biases['decoder_b3']))
    pred = tf.nn.sigmoid(tf.add(tf.matmul(decoder_h3, weights['decoder_h4']), biases['decoder_b4']))

    '''
    设置代价函数
    '''
    # 求平均
    cost = tf.reduce_mean((input_y - pred) ** 2)

    '''
    求解,开始训练
    '''
    train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 计算一轮跌倒多少次
        num_batch = int(np.ceil(mnist.train.num_examples / batch_size))

        # 迭代
        for epoch in range(training_epochs):

            sum_loss = 0.0
            for i in range(num_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                _, loss = sess.run([train, cost], feed_dict={input_x: batch_x})
                sum_loss += loss

            # 打印信息
            if epoch % display_epoch == 0:
                print('Epoch {}  cost = {:.9f}'.format(epoch + 1, sum_loss / num_batch))
        print('训练完成')

        # 输出图像数据最大值和最小值
        print('最大值：', np.max(mnist.train.images[0]), '最小值:', np.min(mnist.train.images[0]))

        '''
        可视化结果
        '''
        reconstruction = sess.run(pred, feed_dict={input_x: mnist.test.images[:show_num]})
        plt.figure(figsize=(1.0 * show_num, 1 * 2))
        for i in range(show_num):
            plt.subplot(2, show_num, i + 1)
            plt.imshow(np.reshape(mnist.test.images[i], (28, 28)), cmap='gray')
            plt.axis('off')
            plt.subplot(2, show_num, i + show_num + 1)
            plt.imshow(np.reshape(reconstruction[i], (28, 28)), cmap='gray')
            plt.axis('off')
        plt.show()

        '''
        显示二维的特征数据  有点聚类的感觉，一般来说通过自编码网络将数据降维之后更有利于进行分类处理
        '''
        plt.figure(figsize=(10, 8))
        # 将onehot转为一维编码
        labels = [np.argmax(y) for y in mnist.test.labels]
        encoder_result = sess.run(encoder_h4, feed_dict={input_x: mnist.test.images})
        plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=labels)
        plt.colorbar()
        plt.show()