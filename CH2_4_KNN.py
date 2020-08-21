#!usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  # K节临近法
from sklearn.metrics import classification_report

'''K近邻算法对生物物种进行分类'''
# 导入数据集
iris = load_iris()
print(iris.data.shape)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)

# 标准化。
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 类别预测
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
y_predict = knc.predict(X_test)

# 准确性测评
print('The accuracy of K-Nearest Neighbor Classifier is', knc.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=iris.target_names))