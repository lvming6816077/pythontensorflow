# 让我们生成一些近似线性的数据（）来测试一下这个方程。

import numpy as np
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)


X_new = np.array([[0],[2]])
# X_new_b = np.c_[np.ones((2, 1)), X_new]
# y_predict = X_new_b.dot(theta_best)

# 查看图片
# import matplotlib.pyplot as plt
# plt.plot(X_new,y_predict,"r-")
# plt.plot(X,y,"b.")
# plt.axis([0,2,0,15])
# plt.show()


# 使用下面的 Scikit-Learn 代码可以达到相同的效果：
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

print (lin_reg.predict(X_new))

# 随机梯度下降
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X,y.ravel())
