# -*- coding:utf-8 -*-

#从sklearn.datasets中导入新闻数据
from sklearn.datasets import fetch_20newsgroups
#与之前预存的数据不同，fetch_20newgroups需要即时从互联网下载数据
news = fetch_20newsgroups(subset = 'all')
#检查数据规模和细节
print len(news.data)
print news.data[0]

#从sklearn.cross_validation中导入train_test_split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state = 33)
print y_train.shape
print y_test.shape

#从skleran.feature_extraction.text里导入用于文本特征向量转化模块。
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

#从sklearn.naive_bayes里导入朴素贝叶斯模型
#multinomialNB:多项式贝叶斯
from sklearn.naive_bayes import MultinomialNB
#从使用默认配置初始化朴素贝叶斯模型
mnb = MultinomialNB()
#利用训练数据对模型参数估计
mnb.fit(X_train, y_train)
#对测试样本进行类别预测，结果存储在变量y_predict中
y_predict = mnb.predict(X_test)

#从sklearn.metrics里导入classification_report用于详细的分类性能报告
from sklearn.metrics import classification_report
print "The accuracy of Naive Bayes Classifier is", mnb.score(X_test, y_test)
print classification_report(y_test, y_predict, target_names = news.target_names)
print target_names
