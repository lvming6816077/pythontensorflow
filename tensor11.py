import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # 插入数据
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# # name在保存模型时非常有用
# x = tf.placeholder("float", [None, 784], name='x') 
# W = tf.Variable(tf.zeros([784, 10]), name='W')  
# b = tf.Variable(tf.zeros([10]), name='b')
# y = tf.nn.softmax(tf.matmul(x, W) + b, name='y')  # y预测概率分布

# y_ = tf.placeholder("float", [None, 10])  # y_实际概率分布

# cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # 交叉熵
# # 梯度下降算法以0.01学习率最小化交叉熵
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  
# init = tf.initialize_all_variables()  # 初始化变量

# sess = tf.Session()
# sess.run(init)
# saver = tf.train.Saver()

# for i in range(1000):  # 开始训练模型，循环1000次
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# saver.save(sess, './tensor11/minst_model.ckpt')  # 保存模型

#https://blog.csdn.net/u011389706/article/details/81223784

from PIL import Image, ImageFilter
import tensorflow as tf

def imageprepare():
    file_name = './5.png'  # 图片路径
    myimage = Image.open(file_name).convert('L')  # 转换成灰度图
    tv = list(myimage.getdata())  # 获取像素值
    # 转换像素范围到[0 1], 0是纯白 1是纯黑
    tva = [(255-x)*1.0/255.0 for x in tv] 
    
    return tva

result = imageprepare()
init = tf.global_variables_initializer()
saver = tf.train.Saver

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.import_meta_graph('./tensor11/minst_model.ckpt.meta')  # 载入模型结构
    saver.restore(sess,  './tensor11/minst_model.ckpt')  # 载入模型参数

    graph = tf.get_default_graph()  # 计算图
    x = graph.get_tensor_by_name("x:0")  # 从模型中获取张量x
    y = graph.get_tensor_by_name("y:0")  # 从模型中获取张量y

    prediction = tf.argmax(y, 1)
    predint = prediction.eval(feed_dict={x: [result]}, session=sess)
    print(predint[0])