# -*- coding:utf-8 -*-
# 导入pandas与numpy工具包
import pandas as pd
import numpy as np

column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

#使用pandas.read_csv函数从互联网读取指定数据
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names = column_names)

#将？替换为标准缺失值
data = data.replace(to_replace = '?', value = np.nan)
#丢弃带有缺失值的数据
data = data.dropna(how = 'any')
#输出data的数据量和维度
print "data shape", data.shape

#使用skleran,cross_valiadtion里的train_test_split模块用于分割数据
from sklearn.cross_validation import train_test_split
#随机采样25%的数据用于测试，剩下的75%用于构建训练集合
#其中train的第一项为划分的数据集，第二项为划分的样本结果，第三项为样本比，第四项为随机种子
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size = 0.25, random_state = 33)

#查验训练样本的数量和类别分布
print "train sample:", y_train.value_counts()
print "test sample:", y_test.value_counts()


#从sklearn.preprocessing里导入StandardScaler
#LogisticRegression和随机梯度参数
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

#标准化数据，保证每个维度的特征数据方差为1，均值为0，
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#初始化
lr = LogisticRegression()
sgdc = SGDClassifier()
#建立模型
lr.fit(X_train, y_train)
#使用训练好的模型进行预测
lr_y_predict = lr.predict(X_test)
#建立模型
sgdc.fit(X_train, y_train)
sgdc_y_predict = sgdc.predict(X_test)
#从sklearn.metrics里导入classification_report模块
from sklearn.metrics import classification_report

#使用logistic regression自带的评分函数score获得模型在测试集上的准确性结果
print "Accuracy of LR Classifier:", lr.score(X_test, y_test)
#利用classification_report模块获得其他三个指标
print classification_report(y_test, lr_y_predict, target_names = ['Benign', 'Malignant'])

#使用随机梯度下降模型
print "Accuracy of SGD Classifier:", sgdc.score(X_test, y_test)
#利用classification_report模块获得其他三个指标
print classification_report(y_test, sgdc_y_predict, target_names = ['Benign', 'Malignant'])
