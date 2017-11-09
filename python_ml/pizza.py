# -*- coding:utf-8 -*-

#输入训练样本的特征以及目标值，分别存储在变量X_train与y_train之中
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

#从sklearn.linear_model导入LinearRegression
from sklearn.linear_model import LinearRegression
#使用默认配置初始化线性回归模型
lr = LinearRegression()
#直接以X_train训练
lr.fit(X_train, y_train)

#导入numpy包处理
import numpy as np
#在x轴上从0-25均匀采样100个数据点
xx = np.linspace(0, 26, 100)
#使之成为（100， 1）
xx = xx.reshape(xx.shape[0], 1)
#以上述100个点作为基准，预测回归直线
yy = lr.predict(xx)

#对回归预测到的直线进行作图
import matplotlib.pyplot as plt
#plt.scatter(X_train, y_train)

#???????????
#plt1, = plt.plot(xx, yy, label = "Degree = 1")
#plt.axis = ([0, 25, 0, 25])
#plt.xlabel('Diameter of Pizza')
#plt.ylabel('Price of Pizza')
#plt.legend(handles = [plt1])
#plt.show()

#输出线性回归模型的评测值
print 'The R-squared value of Linear Regression performing on the training data is', lr.score(X_train, y_train)

#从sklearn.preprocessing中导入多项式特征产生器
from sklearn.preprocessing import PolynomialFeatures
#使用PolynomialFeatures映射出2次多项式特征，存储在变量X_train_poly2中
poly2 = PolynomialFeatures(degree = 2)
X_train_poly2 = poly2.fit_transform(X_train)

#以线性回归器为基础，初始化回归模型。尽管特征的维度有提升，但是模型基础仍然使线性模型
lr_poly2 = LinearRegression()

#对2次多项式回归模型进行训练
lr_poly2.fit(X_train_poly2, y_train)

#从新映射绘图用x轴采样数据
xx_poly2 = poly2.transform(xx)

#使用2次多项式回归模型对应x轴采样数据进行回归预测
yy_poly2 = lr_poly2.predict(xx_poly2)

#分别对训练数据点、线性回归直线、2次多项式回归曲线进行作图
#plt.scatter(X_train, y_train)

#plt1 ,= plt.plot(xx, yy, label = 'Degree = 1')
#plt2 ,= plt.plot(xx, yy_poly2, label = 'Degree = 2')

#plt.axis([0, 25, 0, 25])
#plt.xlabel('Diameter of Pizza')
#plt.ylabel('Price of Pizza')
#plt.legend(handles = [plt1, plt2])
#plt.show()

#输出2次多项式回归模型在训练样本上的R-squared值
print "The R-squared value of Polynomial Regressor performing on the training data is",lr_poly2.score(X_train_poly2, y_train)

#使用4次多项式特征生成器
poly4 = PolynomialFeatures(degree = 4)
X_train_poly4 = poly4.fit_transform(X_train)

#使用默认配置初始化
lr_poly4 = LinearRegression()
#对模型进行训练
lr_poly4.fit(X_train_poly4, y_train)

#从新映射图用x轴采样数据
xx_poly4 = poly4.transform(xx)
#使用4次多项式回归模型对应x轴采样数据进行回归预测
yy_poly4 = lr_poly4.predict(xx_poly4)

#分别对训练数据点，线性回归直线，2次多项式以及4次多项式回归曲线进行作图
plt.scatter(X_train, y_train)
plt1 ,= plt.plot(xx, yy, label = 'Degree = 1')
plt2 ,= plt.plot(xx, yy_poly2, label = 'Degree = 2')
plt4 ,= plt.plot(xx, yy_poly4, label = 'Degree = 4')

plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles = [plt1, plt2, plt4])
plt.show()

print 'The R-squared value of Polynomial Regressor perform on the training data is', lr_poly4.score(X_train_poly4, y_train)

#准备测试数据
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

#使用测试数据对线性回归模型的性能进行评估
print 'The R-squared value on test:', lr.score(X_test, y_test)

#使用测试数据对2次多项式回归模型的性能进行评估
X_test_poly2 = poly2.transform(X_test)
print 'The R-squared value on test(poly2):', lr_poly2.score(X_test_poly2, y_test)

#使用测试数据对4次多项式回归模型的性能进行评估
X_test_poly4 = poly4.transform(X_test)
print 'The R-squared value on test(poly4):', lr_poly4.score(X_test_poly4, y_test)

#从sklearn.linear_model导入Lasso
from sklearn.linear_model import Lasso
#使用默认配置初始化
lasso_poly4 = Lasso()
#使用Lasso对4次多项式特征进行拟合
lasso_poly4.fit(X_train_poly4, y_train)
#对lasso模型进行评估
print lasso_poly4.score(X_test_poly4, y_test)

#输出Lasso模型的参数列表
print lasso_poly4.coef_

#输出普通4次函数的性能和参数列表
print lr_poly4.score(X_test_poly4, y_test)
print lr_poly4.coef_

#输出上述参数的平方和，验证参数之间的巨大差异
print np.sum(lr_poly4.coef_ ** 2)

#从sklearn.linear_model导入L2范数，Ridge
from sklearn.linear_model import Ridge
#使用默认初始化
ridge_poly4 = Ridge()

#使用ridge进行拟合
ridge_poly4.fit(X_train_poly4, y_train)

#输出回归的性能
print ridge_poly4.score(X_test_poly4, y_test)
#输出相关系数
print ridge_poly4.coef_

#计算平方和，验证参数差异
print np.sum(ridge_poly4.coef_ ** 2)









































