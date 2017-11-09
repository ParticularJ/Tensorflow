# -*- coding:utf-8 -*-
#导入pandas用于数据分析
import pandas as pd
#利用pandas的read_csv模块直接从互联网读取
Titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
#观察前几行数据，可以发现，数据种类各异，数值型、类别型，还有缺失
print Titanic.head()
#使用pandas,数据都转入pandas独有的dataframe格式（二维数据表格），直接使用info()，产看数据统计特性
Titanic.info()


#选取合适的特征
X = Titanic[['pclass', 'age', 'sex']]
y = Titanic['survived']

#对当前任务进行查看
X.info()

#借由上面的输出，我们设计如下几个数据处理的任务：
# 1）age这个数据列，只有633个，需要补充
# 2）sex与pclass两个数据列的值都是类别型，需要转换为数值特征，用0/1代替

# 补充age中的数据，使用平均数或者中位数都是对模型偏离造成最小影响的策略

X['age'].fillna(X['age'].mean(), inplace = True)

#对补充的数据进行查看
X.info()

#数据分割
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 33)

#使用scikit-learn.feature_extraction中的特征转换器
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = False)

#转换特征后，我们发现凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变
X_train = vec.fit_transform(X_train.to_dict(orient = 'record'))
print vec.feature_names_

#同样需要对测试数据的特征进行转换
X_test = vec.transform(X_test.to_dict(orient = 'record'))

#从sklearn.tree中导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
#使用默认配置初始化决策树分类器
dtc = DecisionTreeClassifier()
#使用分割到的训练数据进行模型训练
dtc.fit(X_train, y_train)
#用训练好的模型进行预测
y_predict = dtc.predict(X_test)

#从sklearn.metrics中导入classification_report
from sklearn.metrics import classification_report
#输出预测准确性
print dtc.score(X_test, y_test)
#输出更加详细的分类功能
print classification_report(y_predict, y_test, target_names = ['died', 'survived'])


#使用随机森林分类器进行集成模型的训练以及分析
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)


#使用梯度提升决策树进行集成模型的训练以及预测分析
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_y_pred = gbc.predict(X_test)


#输出随机森林在测试集上的指标
print 'The accuracy of random forest classifier is', rfc.score(X_test, y_test)
print classification_report(rfc_y_pred, y_test)

#输出梯度提升决策树在测试集上的分类准确性等指标
print 'The accuracy of gradient boosting classifier is', gbc.score(X_test, y_test)
print classification_report(gbc_y_pred, y_test)
