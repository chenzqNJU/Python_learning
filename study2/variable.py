import tensorflow as tf
v = tf.Variable(3, name='v')
v2 = v.assign(5)
sess = tf.InteractiveSession()
sess.run(v.initializer)
sess.run(v)
3
sess.run(v2)
5

#初始化
v1.initializer.run()#使用初始化器 initializer op 的 run() 方法初始化 'biases'
sess.run(tf.global_variables_initializer())
sess.run(v1.initializer)







import tensorflow as tf
init = tf.constant_initializer([5])
x = tf.get_variable('x', shape=[1], initializer=init)
sess = tf.InteractiveSession()
sess.run(x.initializer)
sess.run(x)

for i in range(4):
    with tf.variable_scope('scope-{}'.format(i)):
        for j in range(25):
             v = tf.Variable(1, name=str(j))
#报错
with tf.variable_scope('scope'):
    v1 = tf.get_variable('var', [1])
    v2 = tf.get_variable('var', [1])
import tensorflow as tf
with tf.variable_scope('scope'):
    v1 = tf.Variable(1, name='var')
    v2 = tf.Variable(2, name='var')
