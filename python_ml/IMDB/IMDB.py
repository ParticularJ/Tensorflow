# -*- coding:utf-8 -*-
#导入pandas包
import pandas as pd

#从本地读入训练和测试数据
train = pd.read_csv('labeledTrainData.tsv', delimiter = '\t')
test = pd.read_csv('testData.tsv', delimiter = '\t')

#检查数据
print train.head(),test.head()

#从bs4导入beautifulSoup用于整洁原始文本
from bs4 import BeautifulSoup
#导入正则表达式工具包
import re

#从nltk.corpus里导入停用词列表
from nltk.corpus import stopwords

#定义review_to_text函数，完成对原始评论的三项数据处理任务
def review_to_text(review, remove_stopwords):
    #去掉html标记
    raw_text = BeautifulSoup(review, 'html').get_text()
    #去掉非字母字符
    letters = re.sub('[^a-zA-Z]', ' ', raw_text)
    words = letters.lower().split()
    #如果remove_stopwords被激活，则进一步去掉评论中的停用词
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    #返回每条评论经此三项预处理任务的词汇列表
    return words

#分别对原始训练和测试数据集进行上述三项预处理
X_train = []
for review in train['review']:
    X_train.append(''.join(review_to_text(review, True)))
X_test = []
for review in test['review']:
    X_test.append(''.join(review_to_text(review, True)))
y_train = train['sentiment']

#导入文本特征抽取器CountVectorizer与TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#从skleran导入贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
#导入Pipeline用于方便搭建朴素贝叶斯系统
from sklearn.pipeline import Pipeline
#导入GridSearch
from sklearn.grid_search import GridSearchCV
#使用pipeline搭建两组使用朴素贝叶斯模型的分类器，区别在于分别使用CountVectorizer和TfidfVectorizer
pip_count = Pipeline([('count_vec', CountVectorizer(analyzer = 'word')), ('mnb', MultinomialNB())])
pip_tfidf = Pipeline([('tfidf_vec', TfidfVectorizer(analyzer = 'word')), ('mnb', MultinomialNB())])

#分别配置用于模型超参数搜索的组合
params_count = {'count_vec__binary': [True, False], 'count_vec__ngram_range': [(1, 1), (1, 2)],
                'mnb__alpha': [0.1, 1.0, 10.0]}
params_tfidf = {'tfidf_vec__binary': [True, False], 'tfidf_vec__ngram_range': [(1, 1), (1, 2)],
                'mnb__alpha': [0.1, 1.0, 10.0]}

#采用4折交叉验证的方法对使用CountVectorizer的朴素贝叶斯模型并行化超参数搜索
gs_count = GridSearchCV(pip_count, params_count, cv = 4, n_jobs = -1, verbose = 1)
gs_count.fit(X_train, y_train)

#输出交叉验证中最佳的准确性得分以及超参数组合
print gs_count.best_score_
print gs_count.best_params_

#以最佳的超参数组合配置模型并对测试数据进行预测
count_y_predict = gs_count.predict(X_test)

#采用4折交叉验证的方法对使用TfidfVectorizer的朴素贝叶斯模型进行并行化超参数搜索
gs_tfidf = GridSearchCV(pip_tfidf, params_tfidf, cv = 4, n_jobs = -1, verbose = 1)
gs_tfidf.fit(X_train, y_train)
#输出最佳的参数
print gs_tfidf.best_score_
print gs_tfidf.best_params_
#以最佳参数组合配置模型并进行预测
tfidf_y_predict = gs_tfidf.predict(X_test)

#使用pandas对需要提交的数据进行格式化
submission_count = pd.DataFrame({'id': test['id'], 'sentiment': count_y_predict})
submission_tfidf = pd.DataFrame({'id': test['id'], 'sentiment': tfidf_y_predict})

#结果输出都本地
submission_count.to_csv('~/mypython/python_ml/IMDB/submission_count.csv', index = False)
submission_tfidf.to_csv('~/mypython/python_ml/IMDB/submission_tfidf.csv', index = False)

#从本地读取未标记数据
unlabeled_train = pd.read_csv('unlabeledTrainData.tsv', delimiter = '\t', quoting = 3)

import nltk.data

#准备使用nltk的tokenizer对影评中的英文句子进行分割
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#定义函数review_to_sentences逐条影评进行分局
def review_to_sentences(review, tokenizer):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_text(raw_sentence, False))
    return sentences

corpora = []
#准备用于训练词向量的数据
for review in unlabeled_train['review']:
    corpora += review_to_sentences(review.decode('utf-8'), tokenizer)

#配置训练词向量模型的超参数
num_features = 300
min_word_count = 20
num_workers = 4
context = 10
downsampling = 1e-3

#从gensim.models导入word2vec
from gensim.models import word2vec
#开始词向量模型的训练
model = word2vec.Word2Vec(corpora, workers = num_workers, size = num_features,
                          min_count = min_word_count, window = context, sample = downsampling)

model.init_sims(replace = True)

model_name = "~/mypython/python_ml/IMDB/300features_20minwords_10context"
#可以将词向量模型的训练结果保存
model.save(model_name)

#直接读入已经训练好的词向量模型
from gensim.models import Word2Vec
model = Word2Vec.load("~/mypython/python_ml/IMDB/300features_20minwords_10context")

#探查一下该词向量模型的训练成果
model.most_similar('man')


import numpy as np 

#定义一个函数使用词向量产生文本特征向量
def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features), dtype = "float32")
    nwords = 0.
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featrueVec, model[word])
    featureVec = np.divide(featureVec, nwords)
    return featureVec

#定义另个每条影评转化为基于词向量的特征向量
def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype = "float32")

    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter += 1
    return reviewFeatureVecs
    
#准备新的基于词向量表示的训练和测试特征向量
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_text(review, remove_stopwords = True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_text(review, remove_stopwords = True))

testDataVecs = getAvgFeatureVecs(clean_test+reviews, model, num_features)

#从sklearn.ensemble导入GradientBoostingClassifier模型进行影评情感分析
from skleran.ensemble import GradientBoostingClassifier

#从skleran.grid_search 导入 GridSearchCV
from sklearn.grid_search import GridSearchCV

gbc = GradientBoostingClassifier()

#配置超参数的搜索组合
params_gbc = {'n_estimators': [10, 100, 500], 'learning_rate': [0.01, 0.1, 1.0],
                'max_depth':  [2, 3, 4]}
gs = GridSearchCV(gbc, params_gbc, cv = 4, n_jobs = -1, verbose = 1)

gs.fit(X_train, y_train)

#输出网络搜索得到的最佳性能
print gs.best_score_
print gs.best_params_

#使用超参数调优之后的梯度上升树模型进行预测
result = gs.predict(testDataVecs)
output = pd.DataFrame(data = {"id": test["id"], "sentiment": result})
output.to_csv("~/mypython/python_ml/IMDB/submission_w2v.csv", index = False, quoting = 3)































































































































