# -*- coding:utf-8 -*-

#分别导入numpy,matplotlib以及pandas,用于数学计算，作图以及数据分析
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#使用pandas.read_csv分别读取训练数据与测试数据集
digits_train = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header = None)
digits_test = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header = None)

#digits_train.info()
#从训练与测试集上都分离出64维度的像素特征与1维度的数字目标
X_train = digits_train[np.arange(64)]
y_train = digits_train[64]

X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

#从sklearn.cluster中导入KMeans模型
from sklearn.cluster import KMeans

#初始化KMeans模型，并设置聚类中心数量为10
kmeans = KMeans(n_clusters = 10)
kmeans.fit(X_train, y_train)
#逐条判断每个测试图像所属的聚类中心
kmeans_y_pred = kmeans.predict(X_test)

#从sklearn导入metrics
from sklearn import metrics
#使用ARI进行Kmeans聚类性能评估
print metrics.adjusted_rand_score(y_test, kmeans_y_pred)






























