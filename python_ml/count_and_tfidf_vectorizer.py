# -*- coding:utf-8 -*-

#从sklearn.datasets里导入20类新闻文本数据抓取器
from sklearn.datasets import fetch_20newsgroups
#从互联网上即时下载新闻样本，subset='all'参数代表下载全部文本
news = fetch_20newsgroups(subset = 'all')

#从sklearn.cross_validation导入train_test_split
from sklearn.cross_validation import train_test_split
#分别75%训练，25测试
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, 
                                                    test_size = 0.25, random_state = 33)

#从sklearn.feature_extraction.text导入CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#采用默认的配置对CountVectorizer进行初始化（默认配置不去除英文停用词），并且赋值给变量count_vec
count_vec = CountVectorizer()

#只使用词频统计的方式将原始训练和测试文本转化为特征向量
X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)

#从sklearn.naive_bayes里导入朴素贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
#使用默认的配置对分类器进行初始化
mnb_count = MultinomialNB()
#使用朴素贝叶斯分类器，对countvectorizer后的训练样本进行学习
mnb_count.fit(X_count_train, y_train)

#输出模型准确性的结果
print 'The accuracy of classifying 20newsgroups using Naive Bayes:', mnb_count.score(X_count_test, y_test)

#将分类结果存储
mnb_y_pred = mnb_count.predict(X_count_test)
#从sklern.metrics导入classification_report
from sklearn.metrics import classification_report
#输出更加详细的评价指标
print classification_report(y_test, mnb_y_pred, target_names = news.target_names)

#从sklearn.feature_extraction.text导入TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#采用默认的配置对TfidfVectorizer进行初始化，并且赋值
tfidf_vec = TfidfVectorizer()

#使用tfidf的方式，将原始训练和测试文本转化为特征向量
X_tfidf_train = tfidf_vec.fit_transform(X_train)
X_tfidf_test = tfidf_vec.transform(X_test)

#依然使用默认配置的朴素贝叶斯分类器，在相同的训练和测试集上，对新的特征量化方式进行评估
mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(X_tfidf_train, y_train)
print 'The accuracy of classifying 20newsgroups with Naive Bayes:', mnb_tfidf.score(X_tfidf_test, y_test)
y_tfidf_pred = mnb_tfidf.predict(X_tfidf_test)
print classification_report(y_test, y_tfidf_pred, target_names = news.target_names)

#使用停用词过滤配置初始化CountVectorizer与TfidfVectorizer
count_filter_vec, tfidf_filter_vec = CountVectorizer(analyzer = 'word', stop_words ='english'), TfidfVectorizer(analyzer = 'word', stop_words = 'english')

#使用带有停用词过滤的CountVectorizer对训练和测试文本分别进行量化处理
X_count_filter_train = count_filter_vec.fit_transform(X_train)
X_count_filter_test = count_filter_vec.transform(X_test)

#使用带有停用词过滤的TfidfVectorizer对训练和测试文本分别进行量化处理
X_tfidf_filter_train = tfidf_filter_vec.fit_transform(X_train)
X_tfidf_filter_test = tfidf_filter_vec.transform(X_test)

#初始化默认配置的朴素贝叶斯分类器，并对CountVectorizer后的数据进行预测与准确性评估
mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(X_count_filter_train, y_train)
print 'The accuracy of classifying 20newsgroups using Naive Bayes:', mnb_count_filter.score(X_count_filter_test, y_test)
y_count_filter_pred = mnb_count_filter.predict(X_count_filter_test)

#初始化另一个默认配置的朴素贝叶斯分类器，并对TfidfVectorizer后的数据进行预测与准确评估性
mnb_tfidf_filter = MultinomialNB()
mnb_tfidf_filter.fit(X_tfidf_filter_train, y_train)
print 'The accuracy of classifying 20newsgroups with Naive Bayes:', mnb_tfidf_filter.score(X_tfidf_filter_test, y_test)
y_tfidf_filter_pred = mnb_tfidf_filter.predict(X_tfidf_filter_test)

#对上述两个模型进行评估
from sklearn.metrics import classification_report
print classification_report(y_test, y_count_filter_pred, target_names = news.target_names)
print classification_report(y_test, y_tfidf_filter_pred, target_names = news.target_names)





















































