# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

n_steps = 20
n_inputs = 1
n_neurons = 100
n_layers = 3
n_outputs = 1
# create a time series

t_min, t_max = 0, 30
resolution = 0.1

def time_series(t):
    return t * np.sin(t) / 3 + 2*np.sin(t * 5)

def next_batch(batch_size, n_steps):
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    # :-1,输出除了最后一个的所有元素
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)

X_batch, y_batch = next_batch(1, n_steps)
#print([X_batch[0], y_batch[0]])

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])


# use dorpout
keep_prob = tf.placeholder_with_default(1.0, shape = ())
# outputprojectionwrapper函数的作用：
# 比如我们现在每一次输出有向量为100，但是我们只想要1个
# 在每一个输出上添加一个全连接层，这些全连接层共享W，b

#cell = tf.contrib.rnn.OutputProjectionWrapper(
#    tf.contrib.rnn.BasicRNNCell(num_units = n_neurons, activation = tf.nn.relu),
#    output_size = n_outputs)

'''
Although using an OutputProjectionWrapper is the simplest solution to reduce the
dimensionality of the RNN’s output sequences down to just one value per time step
(per instance), it is not the most efficient. There is a trickier but more efficient 
solution: you can reshape the RNN outputs from [batch_size, n_steps, n_neurons]
to [batch_size * n_steps, n_neurons] , then apply a single fully connected layer
with the appropriate output size (in our case just 1), which will result in an output
tensor of shape [batch_size * n_steps, n_outputs] , and then reshape this tensor
to [batch_size, n_steps, n_outputs] .
'''

from tensorflow.contrib.layers import fully_connected
cells = [tf.contrib.rnn.BasicRNNCell(num_units = n_neurons, activation = tf.nn.relu)
            for layer in range(n_layers)]
cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = keep_prob)
                for cell in cells]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells_drop)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype = tf.float32)
stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = fully_connected(stacked_rnn_outputs, n_outputs, activation_fn = None)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
#outputs, states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)

learning_rate = 0.001

loss = tf.reduce_mean(tf.square(outputs-y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

n_iterations = 1500
batch_size = 50

with tf.Session() as sess:
    sess.run(init)
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict = {X: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict = {X: X_batch, y: y_batch})
            print(iteration, "\tMSE", mse)

    saver.save(sess, "./my_time_series_model")













