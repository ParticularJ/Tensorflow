# -*- coding:utf-8 -*-

#导入pandas包用于数据读取
import pandas as pd
import numpy as np

#从互联网读取手写体数字
digits_train = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header = None)

digits_test = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header = None)

#分割训练数据的特征向量和标记
X_digits = digits_train[np.arange(64)]
y_digits = digits_train[64]

#从sklearn.decomposition导入pca
from sklearn.decomposition import PCA

#初始化一个可以将高纬度特征向量压缩到两个维度的PCA
estimator = PCA(n_components = 2)
X_pca = estimator.fit_transform(X_digits)

#显示10类手写体数字图片经压缩后的2维空间分布
import matplotlib.pyplot as plt

def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime',
             'cyan', 'orange', 'gray']
    for i in xrange(len(colors)):
        px = X_pca[:, 0][y_digits.as_matrix() == i]
        py = X_pca[:, 1][y_digits.as_matrix() == i]
        plt.scatter(px, py, c = colors[i])
    
    #astype将指定
    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()

#plot_pca_scatter()

#对训练数据，测试数据进行特征向量与分类目标的分割
X_train = digits_train[np.arange(64)]
y_train = digits_train[64]
X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

#导入基于线性核的支持向量机分类器
from sklearn.svm import LinearSVC

#使用默认配置初始化LinearSVC,对原始64维像素特征的训练数据进行建模，并在测试数据上做出预测，存储
svc = LinearSVC()
svc.fit(X_train, y_train)
svc_y_pred = svc.predict(X_test)

#使用pca将64压缩成20维
estimator = PCA(n_components = 10)

#利用训练特征决定20个正交维度的方向，并转化原训练特征
pca_X_train = estimator.fit_transform(X_train)
#测试特征也按照上述的20个正交维度方向进行转换
pca_X_test = estimator.transform(X_test)

#使用默认配置初始化LinearSVC,对压缩过后的二十维特征的训练数据进行建模，并在测试数据上预测，
pca_svc = LinearSVC()
pca_svc.fit(pca_X_train, y_train)
pca_svc_y_pred = pca_svc.predict(pca_X_test)

#从skleran.metrics导入classification_report用于更加细致的分类性能指标
from sklearn.metrics import classification_report

#对使用原始数据特征训练的支持向量分类器进行性能评估
print svc.score(X_test, y_test)
print classification_report(y_test, svc_y_pred, target_names = np.arange(10).astype(str))


#对使用降维后的性能进行评估
print pca_svc.score(pca_X_test, y_test)
print classification_report(y_test, pca_svc_y_pred, target_names = np.arange(10).astype(str))













































































