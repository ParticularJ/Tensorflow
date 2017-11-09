# -*- coding:utf-8 -*-

#导入tensorflow
import tensorflow as tf
#导入numpy
import numpy as np
#导入pandas
import pandas as pd

#从本地使用pandas读取乳腺癌肿瘤的训练和测试数据
train = pd.read_csv('breast-cancer-train.csv')
test = pd.read_csv('breast-cancer-test.csv')

#分割特征与分类目标
X_train = np.float32(train[['Clump Thickness', 'Cell Size']].T)
y_train = np.float32(train['Type'].T)
X_test = np.float32(test[['Clump Thickness', 'Cell Size']].T)
y_test = np.float32(test['Type'].T)

#定义一个tensorflow的变量b作为线性模型的截距，同时设置初始值1.0
b = tf.Variable(tf.zeros([1]))
print "b value:",b 
#定义一个tensorflow的变量W作为线性模型的系数，并设置初始值为-1至1之间的均匀分布的随机数
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))

#显示定义这个线性函数
y = tf.matmul(W, X_train) + b 

#使用tensorflow中的reduce_mean取得训练集上均方误差
loss = tf.reduce_mean(tf.square(y - y_train))

#使用梯度下降法估计参数W, b，并且设置迭代步长为0.01，
optimizer = tf.train.GradientDescentOptimizer(0.01)

#以最小二乘为损失为优化目标
train = optimizer.minimize(loss)

#初始化所有变量
init = tf.initialize_all_variables()

#开启Tensorflow中的会话
sess = tf.Session()

#执行变量初始化操作
sess.run(init)

#迭代1000轮次，训练参数
for step in xrange(0, 1000):
    sess.run(train)
    if step % 200 == 0:
        print step, sess.run(W), sess.run(b)

#准备测试样本
test_negative = test.loc[test['Type'] == 0][['Clump Thickness', 'Cell Size']]
test_positive = test.loc[test['Type'] == 1][['Clump Thickness', 'Cell Size']]

#以最终更新的参数作图
import matplotlib.pyplot as plt
plt.scatter(test_negative['Clump Thickness'], test_negative['Cell Size'], marker = 'o', s = 200, c = 'red')
plt.scatter(test_positive['Clump Thickness'], test_positive['Cell Size'], marker = 'x', s = 150, c = 'black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')

lx = np.arange(0, 12)

#以0.5作为分界面，
ly = (0.5 - sess.run(b) - lx * sess.run(W)[0][0])/sess.run(W)[0][1]
print sess.run(W)[0][0],sess.run(W)[0][1]
plt.plot(lx, ly, color = 'green')
plt.show()





























