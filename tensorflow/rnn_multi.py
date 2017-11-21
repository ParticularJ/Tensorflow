# -*- coding:utf-8 -*- 

'''
如果有多个时间序列，我们不可能创建多个inputs和outputs，
也不可能feed多个placeholder,操作多个outputs值
'''

import tensorflow as tf

n_steps = 2
n_inputs = 3
n_neurons = 5

# None:mini-batch size(extracts the list of input sequences for each time step)
# X_seqs is a Python list of n_steps tensors of shape[None ,inputs]
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# transpose的作用是交换X的前两维，让time step 成为第一维。
# perm[0, 1, 2]表示原始的矩阵维度。其中0：最外层，1：外向内数第二维，2：最内层
# perm[1, 0, 2]就是控制0,1交换转置，最外两层。比如说原来为2*3*4， 变换后为3*2*4
# 变成[n_steps, None, n_inputs]
# Then we extract a Python list of tensors along the first dimension unsing unstack()
X_seqs = tf.unstack(tf.transpose(X, perm = [1, 0, 2]))

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons)

output_seqs, state = tf.nn.static_rnn(basic_cell, X_seqs, dtype = tf.float32)

# merge all output tensors into a single tensor
# swap the first two dimensions to get  a final outputs tensor of shape[None ,n_steps, n_neurons]
outputs = tf.transpose(tf.stack(output_seqs), perm = [1, 0, 2])

init = tf.global_variables_initializer()

import numpy as np 

X_batch = np.array([
        # t = 0      t = 1 
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0]], # instance 2
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [3, 2, 1]], # instance 4
    ])

with tf.Session() as sess:
    sess.run(init)
    outputs_val = sess.run(outputs, feed_dict = {X: X_batch})
    print(outputs_val)
