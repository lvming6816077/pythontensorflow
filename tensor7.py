from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original',data_home='./datasets')
# print(mnist)
# print(len(mnist.data[0]))
X, y = mnist["data"], mnist["target"]
# print(len(X[60000:]))



# 查看图片
# import matplotlib.pyplot as plt
# import matplotlib as matplotlib
# some_digit = X[16000]
# some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()
# print(y[16000])

# 拆分训练数据集和测试数据集
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]



# 让我们打乱训练集。这可以保证交叉验证的每一折都是相似（你不会期待某一折缺少某类数字）
import numpy as np

shuffle_index = np.random.permutation(60000)
# print(len(shuffle_index))
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# print (y[26301])
# 这个“数字 5 检测器”就是一个二分类器，能够识别两类别，“是 5”和“非 5”。让我们为这个分类任务创建目标向量：


y_train_5 = (y_train == 5) # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

# print(y_train_5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
# 多类分类
sgd_clf.fit(X_train, y_train)

my_digit = X[26301]

print (sgd_clf.predict([my_digit]))

print (sgd_clf.decision_function([my_digit]))