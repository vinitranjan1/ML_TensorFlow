import tensorflow as tf
node1 = tf.constant(3.0, dtype = tf.float32)
node2 = tf.constant(4.0) #implicitly tf.float32
print(node1,node2)

#Need to open a session in otder to evaluate nodes
sess = tf.Session()
print(sess.run([node1,node2]))
#Can do things with these nodes
node3 = tf.add(node1,node2)
print("node3:",node3)
print("sess.run(node3):", sess.run(node3))
#Need to run the node to actually evaluate it
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
#Essentially a lambda in functional programming

print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b:[2,4]}))
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 4.5}))
#Expressions can take other expressions, just like a lambda

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
#Make a model to use 
#Variables not initialized at first like tf.constant

init = tf.global_variables_initializer()
sess.run(init)
#Need these lines to explicitly initialize variables

print(sess.run(linear_model, {x: [1,2,3,4]}))
#x is just a placeholder, can feed a list into it (R-esque)

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
#reduce sum abstracts error
print(sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]}))
#sum of squares of delta is common loss function

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
#tf.assign changes tf.Variable
sess.run([fixW,fixb])
print(sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]}))
#values are "perfect" because we made them so, need ML to find these
