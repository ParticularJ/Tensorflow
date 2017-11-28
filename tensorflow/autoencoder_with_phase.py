# -*- coding:utf-8 -*-

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0001

import tensorflow as tf
import sys

activation = tf.nn.elu
regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
# 返回一个生成张量，而不缩放方差的初始化器，使用he initializer
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, [None, n_inputs])

weights1_init = initializer([n_inputs, n_hidden1])
weights2_init = initializer([n_hidden1, n_hidden2])
weights3_init = initializer([n_hidden2, n_hidden3])
weights4_init = initializer([n_hidden3, n_outputs])

weights1 = tf.Variable(weights1_init, dtype = tf.float32, name = "weights1")
weights2 = tf.Variable(weights2_init, dtype = tf.float32, name = "weights2")
weights3 = tf.Variable(weights3_init, dtype = tf.float32, name = "weights3")
weights4 = tf.Variable(weights4_init, dtype = tf.float32, name = "weights4")

biases1 = tf.Variable(tf.zeros(n_hidden1), name = "biases1")
biases2 = tf.Variable(tf.zeros(n_hidden2), name = "biases2")
biases3 = tf.Variable(tf.zeros(n_hidden3), name = "biases3")
biases4 = tf.Variable(tf.zeros(n_outputs), name = "biases4")

hidden1 = activation(tf.matmul(X, weights1) + biases1)
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
outputs = tf.matmul(hidden3, weights4) + biases4

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
optimizer = tf.train.AdamOptimizer(learning_rate)


# 分为两个phase，第一个是为了绕过，hidden2,3,使得输出尽可能与输入接近
with tf.name_scope("phase1"):
    phase1_outputs = tf.matmul(hidden1, weights4) + biases4
    phase1_reconstruction_loss = tf.reduce_mean(tf.square(phase1_outputs-X))
    phase1_reg_loss = regularizer(weights1) + regularizer(weights4)
    phase1_loss = tf.add(phase1_reconstruction_loss, phase1_reg_loss)
    phase1_training_op = optimizer.minimize(phase1_loss)


# 使得第一层和第三层的个数相近，因为要使得它尽可能对称，训练中要freeze hidden1
with tf.name_scope("phase2"):
    phase2_reconstruction_loss = tf.reduce_mean(tf.square(hidden3 - hidden1))
    phase2_reg_loss = regularizer(weights2) + regularizer(weights3)
    phase2_loss = phase2_reconstruction_loss + phase2_reg_loss
    train_vars = [weights2, biases2, weights3, biases3]
    # freeze hidden1
    phase2_training_op = optimizer.minimize(phase2_loss, var_list = train_vars) 

init = tf.global_variables_initializer()
saver = tf.train.Saver()

training_op = [phase1_training_op, phase2_training_op]
reconstruction_losses = [phase1_reconstruction_loss, phase2_reconstruction_loss]
n_epochs = [4, 4]
batch_sizes = [150, 150]

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("FLAG.data_dir")


with tf.Session() as sess:
    init.run()
    for phase in range(2):
        # {}和format是连用的，
        print("Training phase #{}".format(phase + 1))
        for epoch in range(n_epochs[phase]):
            n_batches = mnist.train.num_examples // batch_sizes[phase]
            for iteration in range(n_batches):
                # 输出0,1,2,3并以百分数显示
                print("\r{}%".format(100 * iteration // n_batches), end = "")
                # 缓存区，可以每隔1s显示一个数字，而不是等程序显示完一口气显示
                sys.stdout.flush()
                X_batch, y_batch = mnist.train.next_batch(batch_sizes[phase])
                sess.run(training_op[phase], feed_dict = {X: X_batch})
            loss_train = reconstruction_losses[phase].eval(feed_dict = {X: X_batch})
            print("\r{}".format(epoch), "Train MSE:", loss_train)
            saver.save(sess, "./my_model_one_at_a_time.ckpt")
    loss_test = reconstruction_loss.eval(feed_dict = {X: mnist.test.images})
    print("Test MSE:", loss_test)




















