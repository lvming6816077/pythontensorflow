import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


W = tf.Variable([.1], dtype=tf.float32)
b = tf.Variable([-.1], dtype=tf.float32)

x = tf.placeholder(tf.float32)

linear_model = W*x + b

y = tf.placeholder(tf.float32)

loss = tf.reduce_sum(tf.square(linear_model - y))

# sess = tf.Session()

init = tf.global_variables_initializer()


# print(sess.run(linear_model, {x: [1, 2, 3, 6, 8]}))



# 创建一个梯度下降优化器，学习率为0.001
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

# 用两个数组保存训练数据
x_train = [1, 2, 3, 6, 8]
y_train = [4.8, 8.5, 10.4, 21.0, 25.3]
saver = tf.train.Saver()
with tf.Session() as sess:
    
    sess.run(init)
    print(sess.run(loss, {x: [1, 2, 3, 6, 8],y: [4.8, 8.5, 10.4, 21.0, 25.3]}))
    # 训练10000次
    for epoch in range(100000):
        # if epoch % 100 == 0:
            # print("Epoch", epoch, "MSE =", loss.eval())
        sess.run(train, {x: x_train, y: y_train})

    # 打印一下训练后的结果
    print('W:%s b:%s loss:%s' %(sess.run(W),sess.run(b),sess.run(loss, {x: x_train , y: y_train})))

    # 保存模型
    save_path = saver.save(sess,'./my_model_final.ckpt')  