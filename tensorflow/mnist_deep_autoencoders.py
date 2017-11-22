# -*- coding:utf-8 -*-

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.001

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
X = tf.placeholder(tf.float32, [None, n_inputs])

with tf.contrib.framework.arg_scope(
                                    [fully_connected],
                                    activation_fn = tf.nn.relu,
                                    weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
                                    weights_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)):
    hidden1 = fully_connected(X, n_hidden1)
    hidden2 = fully_connected(hidden1, n_hidden2)
    hidden3 = fully_connected(hidden2, n_hidden3)
    outputs = fully_connected(hidden3, n_outputs, activation_fn = None)

reconstruction_loss = tf.reduce_mean(tf.square(X-outputs))

#
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
        
n_epochs = 5
batch_size = 150

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("FLAG.data_dir")

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict = {X: X_batch})
        loss_train = loss.eval(feed_dict = {X: X_batch})
        print(epoch, "Train loss: ", loss_train)




















