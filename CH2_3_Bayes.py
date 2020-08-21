#!usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

'''朴素贝叶斯对20类新闻文本数据进行类别预测'''

# 即使从互联网上下载数据
news=fetch_20newsgroups(subset='all')
# 数据规模
print len(news.data)
# 数据样本细节
print news.data[0]

# 分割测试数据
x_train,x_test,y_train,y_test=train_test_split(news.data,news.target,test_size=0.25,random_state=33)

# 文本特征向量转化模块
vec=CountVectorizer()
x_train=vec.fit_transform(x_train)
x_test=vec.transform(x_test)

# 朴素贝叶斯模型
mnb=MultinomialNB()
mnb.fit(x_train,y_train)
print x_test.shape
print x_train.shape
y_predict=mnb.predict(x_test)
print 'The Accuracy of Naive Bayes Classifier is', mnb.score(x_test,y_test)
print classification_report(y_test,y_predict,target_names=news.target_names)

print "done"

