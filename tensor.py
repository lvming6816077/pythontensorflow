import tensorflow as tf


# node1 = tf.constant(3.2)
# node2 = tf.constant(4.8)

# adder = node1 + node2

# sess = tf.Session()
# print(sess.run(adder))


W = tf.Variable([.1], dtype=tf.float32)
b = tf.Variable([-.1], dtype=tf.float32)

