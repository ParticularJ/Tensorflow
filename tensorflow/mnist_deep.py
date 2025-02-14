# MNIST with softmax

# load the data

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import argparse
import sys
import tempfile
from tensorflow.examples.tutorials.mnist import input_data



FLAGS = None
import tensorflow as tf

def deepnn(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])


    with tf.name_scope('conv1'):
    # First Convolutional Layer
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])
    
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout Layer
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob
# Train and Evaluate 


# Weight Initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# Convolution and Pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1],
                          strides = [1, 2, 2, 1], padding = 'SAME')


def main(_):
    mnist = input_data.read_data_sets("FLAG.data_dir", one_hot = True)
    # create the model    
    x = tf.placeholder(tf.float32, [None, 784])
    # define loss and entropy
    y_ = tf.placeholder(tf.float32, [None, 10])
    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y_conv))
    with tf.name_scope('Adam_Optimizer'):    
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    with tf.name_scope('Accuracy'):       
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    im = cv2.imread('03.bmp', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    im = cv2.resize(im, (28, 28), interpolation=cv2.INTER_CUBIC)
    img_gray = (im - (255 / 2.0)) / 255
    img_gray = np.rot90(img_gray)
    cv2.imshow('out', img_gray)
    cv2.waitKey(0)
    x_img = np.reshape(img_gray, [-1, 784])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict = {
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training acuuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        print(sess.eval(y_conv,feed_dict={x: x_img , keep_prob : 1.0}))
        #save_path = saver.save(sess,"./Model/my_model_final.ckpt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str,
                        default = '/tmp/tensorflow/mnist/input_data',
                        help = 'Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main = main, argv = [sys.argv[0]] + unparsed)




























































