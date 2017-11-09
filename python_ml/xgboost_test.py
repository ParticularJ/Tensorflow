# -*- coding:utf-8 -*-
#导入pandas用于数据处理
import pandas as pd

#通过url下载数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

#选取pclass.age.以及sex作为训练
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']

#对缺失的age信息，采用平均值方法进行补全，
X['age'].fillna(X['age'].mean(), inplace = True)

#对原始数据进行分割，随机采样25%作为测试集
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 33)

#从sklearn.feature_extraction导入DictVectorizer
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = True)

#对原数据进行特征向量化处理
X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
X_test = vec.transform(X_test.to_dict(orient = 'record'))

#采用默认配置的随机森林分类器对测试集进行预测
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
print 'The accuracy of Random Forest Classifier on testing set:', rfc.score(X_test, y_test)

#采用默认配置的XGBoost模型对相同的测试集进行预测
from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)

print 'The accuracy of eXtreme Gradient Boosting Classifier on testing set:', xgbc.score(X_test, y_test)

































