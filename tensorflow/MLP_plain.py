# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs =  10

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
y = tf.placeholder(tf.int32, shape = (None), name = "y")

# create the neuron_layer

def neuron_layer(X, n_neurons, name, activation = None):

# look much nicer in TensorBoard
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        # 产生高斯随机初始化值，可以让suanfa
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev)
        W = tf.Variable(init, name = "weights")
        
        b = tf.Variable(tf.zeros([n_neurons]), name = "biases")
        z = tf.matmul(X, W) + b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z

with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation ="relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation = "relu")
    logits = neuron_layer(hidden2, n_outputs, "outputs")

'''
# tensorflow 自带的一个全连接函数，他默认使用relu，我们使用这个函数就不用自己定义neurons_layer()函数。
from tensorflow.contrib.layers import fully_connected
with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, scope = "hidden1")
    hidden2 = fully_connected(hidden1, n_hedden2, scope = "hidden2")
    logits = fully_connected(hidden2, n_outputs, scope = "outputs", activation_fn = None)
'''

with tf.name_scope("loss"):
    # 这行代码相当于同时使用了softmax函数并且对它求交叉熵
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
    loss = tf.reduce_mean(xentropy, name = "loss")

learning_rate  = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


with tf.name_scope("eval"):
    # 我们需要检查神经网络预测出的最高的logits结果是否和目标相对应，
    # in_top_k() 看真实结果是否在topK的预测结果中， logits表示预测结果， y表示真是结果，1表示top1,返回boolean
    correct = tf.nn.in_top_k(logits, y, 1)
    # 将boolean值转化为float,并且求平均，即1占的百分比
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init  = tf.global_variables_initializer()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("FLAG.data_dir/")

n_epochs = 400
batch_size = 50

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        # // 运算符，做除法，算出整数部分，抛弃余数
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test  = accuracy.eval(feed_dict={X: mnist.test.images, y:mnist.test.labels})
        if epoch % 10 == 0:
            print(epoch, "Train accuracy", acc_train, "Test accuracy", acc_test)






















