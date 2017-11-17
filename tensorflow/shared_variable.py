import tensorflow as tf


with tf.variable_scope("my_scope"):
    x0 = tf.get_variable("x", shape = (), initializer = tf.constant_initializer(0.))
    x1 = tf.Variable(0., name= 'X1')
    x2 = tf.Variable(0., name= 'X2')

# 通过get_variable()共享变量
with tf.variable_scope("my_scope", reuse = True):
    x3 = tf.get_variable("x")
    x4 = tf.Variable(0., name= 'X')

# 同样为共享变量的一种方法
with tf.variable_scope("", default_name = "", reuse = True):
    x5 = tf.get_variable("my_scope/x")

print("x0:", x0.op.name)
print("x1:", x1.op.name)
print("x2:", x2.op.name)
print("x3:", x3.op.name)
print("x4:", x4.op.name)
print("x5:", x5.op.name)

print(x0 is x3 and x3 is x5)
