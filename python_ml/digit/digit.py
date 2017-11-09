# -*- coding:utf-8 -*-

#从sklearn.datasets导入手写体数字加载器
from sklearn.datasets import load_digits
#通过数据加载器获得手写体数字的数码图像数据并储存在digits变量中
digits = load_digits()
#检视数据规模和特征维度
print "digit dimension:", digits.data.shape

#从sklearn.cross_validation中导入train_test_split用于数据分割
from sklearn.cross_validation import train_test_split
#随机选取75%的数据作为训练样本，其余为测试
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, 
                                                    test_size = 0.25, random_state = 33)

#分别检视训练和测试集
print "Train set:", y_train.shape
print "Test set:", y_test.shape

#从sklearn.preprocessing里导入数据标准化模块
from sklearn.preprocessing import StandardScaler
#从sklearn.svm里导入基于线性假设的支持向量机分类器LinearSVC
from sklearn.svm import LinearSVC

#从仍需要对训练和测试的特征数据进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#初始化线性假设的支持向量机分类器LinearSVC
lsvc = LinearSVC()
#进行模型训练
lsvc.fit(X_train, y_train)
#对训练好的模型对测试样本的数字类别进行预测，预测结果存储在变量y_predict中
y_predict = lsvc.predict(X_test)

#使用模型自带的评估函数进行准确性评测
print "The accuracy of linear SVC is:", lsvc.score(X_test, y_test)
#使用sklearn.metric里面的classification_report模块对预测结果做更加详细的分析
from sklearn.metrics import classification_report
print classification_report(y_test, y_predict, target_names = digits.target_names.astype(str))
