# -*- coding:utf-8 -*-
#######   版本问题skflow和tensorflow   
# 导入需要使用的包
from sklearn import datasets, metrics, preprocessing, cross_validation

#使用datasets.load_boston读取数据
boston = datasets.load_boston()

#获取房屋数据特征以及对应房价
X, y = boston.data, boston.target

#分割数据，随机 采样25%
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, random_state = 33, test_size = 0.25)

#对数据进行标准化处理
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#导入skflow
import skflow

#使用skflow的LinearRegressor
tf_lr = skflow.tensorflowlinearregressor(steps = 10000, learning_rate = 0.01, batch_size = 50)
tf_lr.fit(X_train, y_train)
tr_lr_y_predict = tf_lr.predict(X_test)

#输出skflow中LinearRegressor模型的回归性能
print 'The mean absolute error of Tensorflow Linear Regressor on boston dataset is', metrics.mean_absolute_error(tf_lr_y_predict, y_test)

print 'The mean squared error of Tensorflow Linear Regressor on boston dataset is', metrics.mean_squared_error(tf_lr_y_predict, y_test)

print 'The mean R-squared value of Tensorflow Linear Regressor on boston dataset is', metrics.r2_score(tf_lr_y_predict, y_test)

#使用skflow 的DNNRegressor，并且注意其每个隐层特征数量的配置
tf_dnn_regressor = skflow.TensorFlowDNNRegressor(hidden_units = [100, 40],
                    steps = 10000, learning_rate = 0.01, batch_size = 50)
tf_dnn_regressor.fit(X_train, y_train)
tf_dnn_y_regressor_predict = tf_dnn_regressor.predict(X_test)

#输出skflow中DNNRegressor模型的回归性能
print 'The mean absolute error of Tensorflow DNN Regressor on boston dataset is', metrics.mean_absolute_error(tf_dnn_y_predict, y_test)

print 'The mean squared error of Tensorflow DNN Regressor on boston dataset is', metrics.mean_squared_error(tf_dnn_y_predict, y_test)

print 'The mean R-squared value of Tensorflow DNN Regressor on boston dataset is', metrics.r2_score(tf_dnn_y_predict, y_test)


#使用Scikit-learn的RandomForestRegressor
from skleran.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_y_predict = rfr.predict(X_test)

#输出scikit中RandomForestRegressor模型的回归性能
print 'The mean absolute error of Tensorflow Sklearn Random Forest Regressor on boston dataset is', metrics.mean_absolute_error(rfr_y_predict, y_test)

print 'The mean squared error of Tensorflow Sklearn Random Forest Regressor on boston dataset is', metrics.mean_squared_error(rfr_y_predict, y_test)

print 'The mean R-squared value of Tensorflow Sklearn Random Forest Regressor on boston dataset is', metrics.r2_score(rfr_y_predict, y_test)



























