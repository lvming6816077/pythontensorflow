
# mnist 逻辑回归
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp/data/")

x = tf.placeholder("float", [None, 784])  # x定义为占位符，待计算图运行的时候才去读取数据图片
W = tf.Variable(tf.zeros([784, 10]))      # 权重w初始化为0
b = tf.Variable(tf.zeros([10]))           # b也初始化为0
y = tf.nn.softmax(tf.matmul(x, W) + b)    # 创建线性模型
y_ = tf.placeholder("float", [None, 10])  # 图片的实际标签，为0~9的数字



# 使用Tensorflow自带的交叉熵函数
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

http://www.cnblogs.com/vipyoumay/p/7507149.html