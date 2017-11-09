#-*- coding:utf-8 -*-
import pandas as pd

#使用csv返回训练地址参数
df_train = pd.read_csv('breast-cancer-train.csv')
#返回测试地址参数
df_test = pd.read_csv('breast-cancer-test.csv')
#选取‘Clump Thickness’与‘Cell Size’作为特征，构建正负样本
df_test_negative = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
df_test_positive = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]

import matplotlib.pyplot as plt

# 绘制良性肿瘤，红
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker = 'o', s = 200, c = 'red')
#恶性肿瘤，蓝
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker = 'x', s = 200, c = 'blue')

#x,y轴说明
plt.xlabel("ClumpThickness")
plt.ylabel("CellSize")
plt.title("Cancer")
plt.show()

import numpy as np
plt.figure()
# 随机采样截距和系数
intercept = np.random.random([1])
coef = np.random.random([2])
lx = np.arange(0,12)
ly = (-intercept - lx * coef[0]) / coef[1]
#绘制直线
plt.plot(lx, ly, c = 'yellow')

plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker = 'o', s = 200, c = 'red') 
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker = 'x', s = 150, c = 'black')

plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.title('Cancer')
plt.show()

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

#使用前10条训练样本学习直线的系数和截距
lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10], df_train['Type'][:10])
print 'Testing accuracy (10 training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type'])


intercept = lr.intercept_
coef = lr.coef_[0, :]
# 原本这个分类面应该是lx * coef[0] + ly * coef[1] + intercept = 0
ly = (-intercept - lx * coef[0]) / coef[1]

plt.plot(lx, ly, c = 'green')
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker = 'o', s = 200, c = 'red') 
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker = 'x', s = 150, c = 'black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.title('Cancer')
plt.show()

#使用全部数据学习
lr = LogisticRegression()
lr.fit(df_train[['Clump Thickness', 'Cell Size']], df_train['Type'])
print 'Testing accuracy (10 training samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type'])

intercept = lr.intercept_
coef = lr.coef_[0, :]
# 原本这个分类面应该是lx * coef[0] + ly * coef[1] + intercept = 0
ly = (-intercept - lx * coef[0]) / coef[1]

plt.plot(lx, ly, c = 'blue')
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker = 'o', s = 200, c = 'red') 
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker = 'x', s = 150, c = 'black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
plt.title('Cancer')
plt.show()
