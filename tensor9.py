# import tensorflow as tf
# dnn 神经网络
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# x = tf.Variable(3, name="x")  
# y = tf.Variable(4, name="y")  
# f = x*x*y + y + 2  

# # way1  
# sess = tf.Session()  
# sess.run(x.initializer)  
# sess.run(y.initializer)  
# result = sess.run(f)  
  
# print(result)  
# sess.close()  

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np

if __name__ == '__main__':
    n_inputs = 28 * 28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    mnist = input_data.read_data_sets("tmp/data/")

    X_train = mnist.train.images
    X_test = mnist.test.images
    y_train = mnist.train.labels.astype("int")
    y_test = mnist.test.labels.astype("int")

    X = tf.placeholder(tf.float32, shape= (None, n_inputs), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name = 'y')

    with tf.name_scope('dnn'):
        hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu
                                  ,name= 'hidden1')

        hidden2 = tf.layers.dense(hidden1, n_hidden2, name='hidden2',
                                  activation= tf.nn.relu)

        logits = tf.layers.dense(hidden2, n_outputs, name='outputs')

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,
                                                                  logits = logits)
        loss = tf.reduce_mean(xentropy, name='loss')#所有值求平均

    learning_rate = 0.01

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits ,y ,1)#是否与真值一致 返回布尔值
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) #tf.cast将数据转化为0,1序列

    init = tf.global_variables_initializer()

    n_epochs = 20
    batch_size = 50
    
    # with tf.Session() as sess:
    #     saver = tf.train.Saver()
    #     init.run()
    #     for epoch in range(n_epochs):
    #         for iteration in range(mnist.train.num_examples // batch_size):
    #             X_batch, y_batch = mnist.train.next_batch(batch_size)
    #             sess.run(training_op,feed_dict={X:X_batch,
    #                                             y: y_batch})
    #         acc_train = accuracy.eval(feed_dict={X:X_batch,
    #                                             y: y_batch})
    #         acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
    #                                             y: mnist.test.labels})
    #         print(X_batch.shape)
    #         print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    
    # #     # saver.restore(sess, "./my_model_final_mnist.ckpt") # or better, use save_path
    #     save_path = saver.save(sess, "./tensor9/my_model_final.ckpt")
    

from PIL import Image, ImageFilter
# import tensorflow as tf

def imageprepare():
    file_name = './5.png'  # 图片路径
    myimage = Image.open(file_name).convert('L')  # 转换成灰度图
    tv = list(myimage.getdata())  # 获取像素值
    # 转换像素范围到[0 1], 0是纯白 1是纯黑
    tva = [(255-x)*1.0/255.0 for x in tv] 
    # print(tva)
    tva = np.array(tva)
    # print(tva)
    return tva

result = imageprepare().reshape(1,784)
print(mnist.test.images.shape)
print(result.reshape(1,784).shape)
# init = tf.global_variables_initializer()
# saver = tf.train.Saver

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.import_meta_graph('./tensor9/my_model_final.ckpt.meta')  # 载入模型结构
    saver.restore(sess,  './tensor9/my_model_final.ckpt')  # 载入模型参数

    y = tf.nn.softmax(y)  # 为了打印出预测值，我们这里增加一步通过softmax函数处理后来输出一个向量
    # y = tf.cast(y, tf.int64)
    # result = sess.run(y, feed_dict={X: result}) 

    # graph = tf.get_default_graph()  # 计算图
#     # x = graph.get_tensor_by_name("x:0")  # 从模型中获取张量x
    # y = graph.get_tensor_by_name("y:0")  # 从模型中获取张量y
    # y = tf.placeholder(tf.int64, shape=(None), name = 'y')
    X = tf.placeholder(tf.float32, shape= (None, n_inputs), name='X')
    # y = 
    prediction = tf.argmax(y, 1)
    predint = prediction.eval(feed_dict={X: result}, session=sess)
    print(result)