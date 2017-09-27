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






