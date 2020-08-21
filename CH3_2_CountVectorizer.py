# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 特征抽取之后使用朴素贝叶斯Navie Bayes分类器进行分类
# 导入20类新闻文本数据抓取器

# 从网上及时下载新闻样本，subset='all'参数代表下载全部近2万条文本存储在变量news中
news = fetch_20newsgroups(subset='all')
X_train,  X_test, y_train, y_test = train_test_split(news.data, news.target,\
                            test_size = 0.25, random_state = 33)
# 默认配置不去出英文停用词
count_vec = CountVectorizer()

"""
如果这里要使用停用词
count_vec = CountVectorizer(analyzer='word', stop_words='english')
"""

# 只使用词频统计的方式将原始训练和测试文本转化为特征向量
X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)

# 导入朴素贝叶斯分类器
mnb_count = MultinomialNB()
# NB进行训练
mnb_count.fit(X_count_train, y_train)

#输出模型的准确性
print('The accuracy of the NB with CountVecorizer:', \
      mnb_count.score(X_count_test, y_test))

# 性能评估
y_predict = mnb_count.predict(X_count_test)
print(classification_report(y_test, y_predict, target_names=news.target_names))