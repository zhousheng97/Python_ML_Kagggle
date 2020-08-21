# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 5组训练数据
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

# 从sklearn.preprocessing导入多项式特征生成器
# 初始化4次多项式特征生成器
poly4 = PolynomialFeatures(degree=4)
X_train_poly4 = poly4.fit_transform(X_train)

regressor_poly4 = LinearRegression()
regressor_poly4.fit(X_train_poly4, y_train)

# 4组测试数据
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]
X_test_poly4 = poly4.transform(X_test)

# 不加入L1范数正则化
print(regressor_poly4.score(X_test_poly4, y_test))
# 回归模型参数列表
print(regressor_poly4.coef_, '\n')
print('求平方和，验证参数之间的差异 \n',np.sum(regressor_poly4.coef_ **2), '\n')

# 加入L1范数正则化
from sklearn.linear_model import Lasso
lasso_poly4 = Lasso()
lasso_poly4.fit(X_train_poly4, y_train)

print(lasso_poly4.score(X_test_poly4, y_test))
# 输出Lasso模型的参数列表
print(lasso_poly4.coef_)

# 加入L2范数正则化
from sklearn.linear_model import Ridge
ridge_poly4 = Ridge()
ridge_poly4.fit(X_train_poly4, y_train)

print(ridge_poly4.score(X_test_poly4, y_test))
print(ridge_poly4.coef_, '\n')
print('观察参数间的差异: \n', np.sum(ridge_poly4.coef_ **2))