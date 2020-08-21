#!usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

'''
比较梯度提升决策树、随机树和单一决策树预测泰坦尼克号幸存人数效果
'''

# 下载数据集
titanic=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
print titanic.head() # 观察前几行数据
print titanic.info() # 查看数据的统计特性

# 特征选择，很可能决定分类的关键特征因素
X = titanic[['Pclass', 'Age', 'Sex']]
y = titanic['Survived']

# 数据处理任务：1.填补缺失数据；2.转化数据特征
X['Age'].fillna(X['Age'].mean(), inplace=True)
# 查看补充完的数据
print X.info()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

# 使用单一决策树训练数据集
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_y_pred = dtc.predict(X_test)

# 使用随机森林训练数据集
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_pred = rfc.predict(X_test)

# 使用梯度提升决策树训练数据集
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_y_pred = gbc.predict(X_test)

# 性能评估
print('-------显示使用决策树预测的结果------------')
print('The accuracy of decision tree is', dtc.score(X_test, y_test))
print(classification_report(dtc_y_pred, y_test))

print('-------显示使用随机树预测的结果------------')
print('The accuracy of random forest classifier is', rfc.score(X_test, y_test))
print(classification_report(rfc_y_pred, y_test))

print('-------显示使用梯度提升决策树预测的结果------------')
print ('The accuracy of gradient tree boosting is', gbc.score(X_test, y_test))
print (classification_report(gbc_y_pred, y_test))