# -*- coding:utf-8 -*-

#导入房价数据
from sklearn.datasets import load_boston
#存储
boston = load_boston()
#输出描述
print boston.DESCR

#导入数据分割器
from sklearn.cross_validation import train_test_split

#导入numpy并重命名
import numpy as np

X = boston.data
y = boston.target

#随机采样25%构建测试样本，其余为训练样本
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 33, test_size = 0.25)


#分析回归目标值的差异
print "The max target value is", np.max(boston.target)
print "The min target value is", np.min(boston.target)
print "The average target value is", np.mean(boston.target)

#从sklearn.preprocessing导入标准化模块
from sklearn.preprocessing import StandardScaler

#分别初始化对特征和目标值的标准化器
ss_X = StandardScaler()
ss_y = StandardScaler()

#分别对训练和测试进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

#从sklearn.linear_model中导入LinearRegression
from sklearn.linear_model import LinearRegression
#使用默认的配置初始化线性回归器
lr = LinearRegression()
#使用参数集进行估计
lr.fit(X_train, y_train)
#预测输出
lr_y_pred = lr.predict(X_test)

#从sklearn.linear_model导入SGDRegressor
from sklearn.linear_model import SGDRegressor
#使用默认配置初始化
sgdr = SGDRegressor()
#使用训练数据进行估计
sgdr.fit(X_train, y_train)
#对测试数据进行预测
sgdr_y_pred = sgdr.predict(X_test)

#使用LinearRegression模型自带的评估模块，并输出评估结果
print 'The value of default measurement of LinearRegression is', lr.score(X_test, y_test)
#从sklearn.metrics依次导入r2_score mean_square_error 以及 mean_absoluate_error用于回归性能评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

#使用r2_score模块，并输出评估结果
print 'The value of R-square of LinearRegression is', r2_score(y_test, lr_y_pred)

#使用mean_squared_error模块，并输出评估结果
print 'The mean squared error of LinearRegression is', mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_pred))

#使用mean_absolute_error模块，并输出评价结果
print 'The mean absolute error of LinearRegression is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_pred))


#使用SGDRegressor模型自带的评估模块，并输出评估结果
print 'The value of default measurement of SGDRegressor is', sgdr.score(X_test, y_test)

#使用r2_score模块，并输出评估结果
print 'The value of R-squared of SGCRegressor is', r2_score(y_test, sgdr_y_pred)

#使用mean_squared_error模块，并输出评估结果
print 'The mean squared error of SGDRegressor is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_pred))

#使用mean_absolute_error模块，并输出评估结果
print 'The mean absoulte error of SGDRegressor is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_pred))


#从sklearn.svm中导入支持向量机回归模型
from sklearn.svm import SVR

#使用线性核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
linear_svr = SVR(kernel = 'linear')
linear_svr.fit(X_train, y_train)
linear_svr_y_pred = linear_svr.predict(X_test)

#使用多项式核函数配置的支持向量进行回归预测
poly_svr = SVR(kernel = 'poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_pred = poly_svr.predict(X_test)

#使用径向基核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
rbf_svr = SVR(kernel = 'rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_pred = rbf_svr.predict(X_test)


#分别使用R-squared,mse,mae指标对三种配置的支持向量回归模型在相同测试集上进行性能评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print 'R-squared value of linear SVR is', linear_svr.score(X_test, y_test)
print 'The mean squared error of linear SVR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_pred))
print 'The mean absolute error of linear SVR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_pred))

print 'R-squared value of poly SVR is', poly_svr.score(X_test, y_test)
print 'The mean squared error of poly SVR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_pred))
print 'The mean absolute error of poly SVR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_pred))

print 'R-squared value of linear SVR is', rbf_svr.score(X_test, y_test)
print 'The mean squared error of rbf SVR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_pred))



#从sklearn.neighbors导入KNeighborRegressor
from sklearn.neighbors import KNeighborsRegressor

#初始化k近邻回归器，并且调整配置，使得预测的方式为平均回归
uni_knr = KNeighborsRegressor(weights = 'uniform')
uni_knr.fit(X_train, y_train)
uni_knr_y_pred = uni_knr.predict(X_test)

#初始化K近邻回归器，并且调整配置，使得预测的方式为根据距离加权回归
dis_knr = KNeighborsRegressor(weights = 'distance')
dis_knr.fit(X_train, y_train)
dis_knr_y_pred = dis_knr.predict(X_test)

#分别使用R-squared,mse,mae指标对三种配置的支持向量回归模型在相同测试集上进行性能评估
print 'R-squared value of uniform knr is', uni_knr.score(X_test, y_test)
print 'The mean squared error of uniform knr is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_pred))
print 'The mean absolute error of uniform knr is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_pred))

print 'R-squared value of distance knr is', dis_knr.score(X_test, y_test)
print 'The mean squared error of distance knr is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_pred))
print 'The mean absolute error of distance knr is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_pred))


#从sklearn.tree中导入DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
#使用默认配置初始化DecisionTreeRegressor
dtr = DecisionTreeRegressor()
#使用波士顿放假的训练数据构建模型
dtr.fit(X_train, y_train)
#s使用默认配置的单一回归树对测试数据进行预测，并存储
dtr_y_pred = dtr.predict(X_test)

#分别使用R-squared,mse,mae指标对三种配置的支持向量回归模型在相同测试集上进行性能评估
print 'R-squared value of DTR is', dtr.score(X_test, y_test)
print 'The mean squared error of DTR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_pred))
print 'The mean absolute error of DTR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_pred))


#从sklearn.ensemble中导入RandomForestRegressor,ExtraTressRgressor以及GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

#使用随机森林训练模型，并预测
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_y_pred = rfr.predict(X_test)

#使用ExtraTreesRegressor训练模型，并预测
etr = ExtraTreesRegressor()
etr.fit(X_train, y_train)
etr_y_pred = etr.predict(X_test)

#使用梯度集成回归
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
gbr_y_pred = gbr.predict(X_test)

#分别使用R-squared,mse,mae指标对三种配置的模型在相同测试集上进行性能评估
print 'R-squared value of RFR is', rfr.score(X_test, y_test)
print 'The mean squared error of RFR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_pred))
print 'The mean absolute error of RFR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_pred))

#分别使用R-squared,mse,mae指标对三种配置的模型在相同测试集上进行性能评估
print 'R-squared value of ETR is', etr.score(X_test, y_test)
print 'The mean squared error of ETR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_pred))
print 'The mean absolute error of ETR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_pred))

#利用训练好的极端回归森林模型，输出每种特征对预测目标的贡献度，axis -1,0 ，1分别代表默认，列，行
print np.sort(zip(etr.feature_importances_, boston.feature_names), axis = 0)

#分别使用R-squared,mse,mae指标对三种配置的支持向量回归模型在相同测试集上进行性能评估
print 'R-squared value of GBR is', gbr.score(X_test, y_test)
print 'The mean squared error of GBR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_pred))
print 'The mean absolute error of GBR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_pred))





















