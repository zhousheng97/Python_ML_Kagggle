#!usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# 加载手写体数据集
digits=load_digits()
# 输出数据规模及维度
print digits.data.shape

# 数据分割
x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)
print y_train.shape

# 标准化
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)

# 初始化线性假设的支持向量机分类器LinearSVC
lsvc=LinearSVC()
# 进行模型训练
lsvc.fit(x_train,y_train)
# 利用训练好的模型对样本进行预测
y_predict=lsvc.predict(x_test)
# 性能测评
print 'The Accuracy of Linear SVC is',lsvc.score(x_test,y_test)
print classification_report(y_test,y_predict,target_names=digits.target_names.astype(str))
