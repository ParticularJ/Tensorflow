# -*- coding:utf-8 -*-

#从sklearn.datasets中导入20类新闻文本抓取器
from sklearn.datasets import fetch_20newsgroups
#导入numpy
import numpy as np

#使用新闻抓取器从互联网上下载所有数据，并且存储在变量news中
news = fetch_20newsgroups(subset = 'all')

#从sklearn.cross_validation 导入train_test_split来分割数据
from sklearn.cross_validation import train_test_split

#对前3000条新闻文本进行数据分割，25%文本用于未来测试
X_train, X_test, y_train, y_test = train_test_split(news.data[:3000], news.target[:3000], test_size = 0.25, random_state = 33)

#导入支持向量机分类模型
from sklearn.svm import SVC
#导入TfidfVectorizer文本抽取器
from sklearn.feature_extraction.text import TfidfVectorizer

#导入Pipeline
from sklearn.pipeline import Pipeline

#使用Pipeline简化系统搭建流程，将文本抽取与分类器模型串联起来。
clf = Pipeline([('vect', TfidfVectorizer(stop_words = 'english', analyzer = 'word')), ('svc', SVC())])

#这里需要实验的2个超参数的个数分别是4、3，svc_gamma的参数共有10^-2, 10^-1... 
#这样我们一共有12种的超参数组合，12个不同参数下的模型
parameters = {'svc__gamma' : np.logspace(-2, 1, 4), 'svc__C' : np.logspace(-1, 1, 3)}

#从sklearn.grid_search中导入网络搜索模块GridSearchCV
from sklearn.grid_search import GridSearchCV

#将12组参数组合以及初始化的Pipeline包括3折交叉验证的要求全部告知GridSearchCV.refit = True设定
gs = GridSearchCV(clf, parameters, verbose = 2, refit = True, cv = 3, n_jobs = -1)

#执行单线程网络搜索
time = gs.fit(X_train, y_train)
gs.best_params_, gs.best_score_

#输出最佳模型在测试集上的准确性
print gs.score(X_test, y_test)
























 
