
# mnist 逻辑回归
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
# from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt # plt 用于显示图片
from PIL import Image, ImageFilter

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

from mnist import MNIST

mndata = MNIST(path='MNIST_data/',gz=True)

x = tf.placeholder(tf.float32, [None, 784])  # x定义为占位符，待计算图运行的时候才去读取数据图片
W = tf.Variable(tf.zeros([784, 10]))      # 权重w初始化为0
b = tf.Variable(tf.zeros([10]))           # b也初始化为0
y = tf.nn.softmax(tf.matmul(x, W) + b)    # 创建线性模型
y_ = tf.placeholder(tf.float32, [None, 10])  # 图片的实际标签，为0~9的数字



# 使用Tensorflow自带的交叉熵函数
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
sess = tf.Session()
# 变量初始化
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(500)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# save_path = saver.save(sess, "./mnistsoftmax/my_model_final.ckpt")

import cv2 
images, labels = mndata.load_testing()
num = 8000
image = images[num]
label = labels[num]
# 打印图片
print(mndata.display(image))
print('这张图片的实际数字是: ' + str(label))

def inverse_color(image):

    height,width = image.shape
    img2 = image.copy()

    for i in range(height):
        for j in range(width):
            img2[i,j] = (255-image[i,j]) 
    return img2
# 测试新图片，并输出预测值
# mp = mpimg.imread('./1.png')

im = Image.open('./1.png').convert('L')

# _im = mnist.train.images[1,:]
# print(_im.shape)
# _im = _im.reshape(28,28)

# print(im.getdata())

tv = list(im.getdata()) #get pixel values

#normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
tva = [ (255-x)*1.0/255.0 for x in tv] 
# print([tva].shape)



# img_gray = cv2.imread('./1.png')
# img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
# mp = tf.cast(mp, tf.float32)
# a = mp.reshape(1, 784)
# cv2.imshow('1',img_gray)
# plt.imshow(img_gray) # 显示图片
# plt.axis('off') # 不显示坐标轴
# plt.show()

# plt.figure()
# plt.imshow(_im)
# plt.show()




# print(mp.reshape(-1).shape)
print(np.array(image).reshape(1, 784).shape)
# print(img_gray)

y = tf.nn.softmax(y)  # 为了打印出预测值，我们这里增加一步通过softmax函数处理后来输出一个向量
result = sess.run(y, feed_dict={x: [tva]})  # result是一个向量，通过索引来判断图片数字
print('预测值为：')
print(result)

