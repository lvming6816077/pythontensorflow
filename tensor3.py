import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


W = tf.Variable([.1], dtype=tf.float32)
b = tf.Variable([-.1], dtype=tf.float32)

x = tf.placeholder(tf.float32)

linear_model = W*x + b

y = tf.placeholder(tf.float32)

loss = tf.reduce_sum(tf.square(linear_model - y))

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

# print(sess.run(linear_model, {x: [1, 2, 3, 6, 8]}))

print(sess.run(loss, {x: [1, 2, 3, 6, 8],y: [4.8, 8.5, 10.4, 21.0, 25.3]}))