# A very simple MNIST classifier

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None

def main(_):
# Import data
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot = True)

# Creat the model
x = tf.placeholder(tf.float32, [None, 784])
w =
