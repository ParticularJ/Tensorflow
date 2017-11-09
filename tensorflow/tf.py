# -*- coding:utf-8 -*-
#导入tensorflow工具包
import tensorflow as tf
#导入numpy
import numpy as np

#初始化一个Tensorflow的常量：hello google tensorflow,
greeting = tf.constant('Hello Google Tensorflow!')

#启动一个会话
sess = tf.Session()
#使用会话执行greeting计算模块
result = sess.run(greeting)
#print result
#sess.close()

#声明matrix1为tensorflow的一个1*2的行向量
matrix1 = tf.constant([[3., 3.]])
#声明matrix2为tensorflow的一个2*1的列向量
matrix2 = tf.constant([[2.], [2.]])

#product将上述两个算子相乘，作为新算例
product = tf.matmul(matrix1, matrix2)

#继续将product与一个标量2.0求和拼接，作为最终的linear算例
linear = tf.add(product, tf.constant(2.0))

#直接在会话中执行linear算例，相当于将上面所有的单独算例拼接成流程图来执行
with tf.Session() as sess:
    results = sess.run(linear)
print results
