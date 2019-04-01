import tensorflow as tf


# node1 = tf.constant(3.2)
# node2 = tf.constant(4.8)

# adder = node1 + node2

# sess = tf.Session()
# print(sess.run(adder))


# W = tf.Variable([.1], dtype=tf.float32)
# b = tf.Variable([-.1], dtype=tf.float32)

#可写函数说明
def printinfo( name, age ):
   #"打印任何传入的字符串"
   print ("Name: ", name)
   print ("Age ", age)
   return
 
#调用printinfo函数
printinfo( age=50, name="miki" )