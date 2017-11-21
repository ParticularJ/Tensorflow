# -*- coding:utf-8 -*-

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_sample_image

# load sample images
china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")

#图片的显示位置
image = china[150:220, 130:250]
#dataset = np.array(load_sample_images().images, dtype = np.float32)
height, width, channels = image.shape
#按照通道求图片的平均值
image_grayscale = image.mean(axis = 2).astype(np.float32)

# 图片的四个通道（mini_batch size, height, width, channels）
images = image_grayscale.reshape(1, height, width, 1)

# Create 2 filters
# shape(filter height, filter width, channels, # filters)
fmap = np.zeros(shape = (7, 7, 1, 2), dtype = np.float32)
# 分别把第三列和第三行赋值为1
fmap[:, 3, :, 0] = 1; #vertical line
fmap[3, :, :, 1] = 1; #horizontal line

def plot_image(image):
    # 图片的插值方法。
    plt.imshow(image, cmap = "gray", interpolation="nearest")
    plt.axis("off")

    
plot_image(fmap[:, :, 0, 0])
plt.show()
plot_image(fmap[:, :, 0, 1])
plt.show()

# Create a graph with input X plus a convolutional layer applying the 2 filters
X = tf.placeholder(tf.float32, shape = (None, height, width, 1))


feature_maps = tf.constant(fmap) 
convolution = tf.nn.conv2d(X, feature_maps, strides = [1, 1, 1, 1], padding = "SAME")

with tf.Session() as sess:
    output = sess.run(convolution, feed_dict = {X: images})

'''
ksize = [batch_size =1 , heighe, width , channels]
max_pool = tf.nn.max_pool(X, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "VALID" )

with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict = {X: images})

plt.imshow(outut[0].astype(np.uint8))
plt.show()
'''

plot_image(images[0, :, :, 0])
plt.show()
# vetical
plot_image(output[0, :, :, 0])
plt.show()
# horizontal
plot_image(output[0, :, :, 1])
plt.show()

