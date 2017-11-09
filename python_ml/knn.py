# -*- coding:utf-8 -*-

#从sklearn.datasets导入iris数据加载器
from sklearn.datasets import load_iris
#使用加载器读取数据并且存入变量
iris = load_iris()

#产看数据的规模
print "The shape of iris:", iris.data.shape
#产看数据说明
print "The description of iris:", iris.DESCR

#从sklearn.cross_validation里选择导入train_test_split用于数据分割
from sklearn.cross_validation import train_test_split
#采样25%
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.25, random_state = 33)

#从sklearn.preprocess里导入数据模块
from sklearn.preprocessing import StandardScaler
#从sklearn.neighborr里选择导入KNN
from sklearn.neighbors import KNeighborsClassifier

#对训练和测试的特征数据进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#使用knn进行预测
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_predict = knc.predict(X_test)

#使用模型自带的评估函数
print 'The accuracy of K-Nearst Neighbor Classifier is', knc.score(X_test, y_test)

#依然使用sklearn.metrics里面的classification_report模块对预测结果做更加详细的分析
from sklearn.metrics import classification_report
print classification_report(y_test, y_predict, target_names = iris.target_names)
