# K-Means with tensorflow for mnist
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
import argparse
import sys
FLAGS = None


# Ignore all GPUs, tf random forest does not benefit from it
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow.examples.tutorials.mnist import input_data
def main(_):
    mnist = input_data.read_data_sets("FLAG.data_dir", one_hot = True)
    full_data_x = mnist.train.images

    #Parameters
    num_steps = 50 #  Total steps to train
    batch_size = 1024 # The number of samples per batch
    k = 25 # The number of clusters
    num_calsses = 10 # The 10 digits
    num_features = 784 # 28*28

    # Input images
    x = tf.placeholder(tf.float32, shape = [None, num_features])
    y = tf.placeholder(tf.float32, shape = [None, num_calsses])

    # Build KMeans graph
    kmeans = KMeans(inputs = x, num_clusters = k, distance_metric = 'cosine',
                    use_mini_batch = True)

    (all_scores,cluster_idx, scores, cluster_centers_initialized, init_op,
    train_op) = kmeans.training_graph()
    cluster_idx = cluster_idx[0] # fix for cluster_idx being a tuple
    avg_distance = tf.reduce_mean(scores)

    init_vars = tf.global_variables_initializer()

    sess = tf.Session()

    sess.run(init_vars, feed_dict = {x: full_data_x})
    sess.run(init_op, feed_dict = {x: full_data_x})

    # Train
    for i in range(1, num_steps+1):
        _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                            feed_dict = {x: full_data_x})
        if i % 10 == 0 or i == 1:
            print("Step %i, Avg Distance: %f" % (i, d))

    # Assign a label to each centroid
    # Count total number of labels per centroid, using the label of each training
    # sample to their closest centroid (given by 'idx')
    counts = np.zeros(shape=(k, num_classes))
    for i in range(len(idx)):
        counts[idx[i]] += mnist.train.labels[i]
    # Assign the most frequent label to the centroid
    labels_map = [np.argmax(c) for c in counts]
    labels_map = tf.convert_to_tensor(labels_map)

    #Evaluation ops
    # Lookup: centroid_id -> label
    cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
    # Compute accuracy
    correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Test Model
    test_x, test_y = mnist.test.images, mnist.test.labels
    print("Test Accuracy:", sess.run(accuracy_op, feed_dict = {x: test_x, y: test_y}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                          help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[1]] + unparsed)















































