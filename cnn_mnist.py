from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    #model function for CNN

    input_layer = tf.reshape(features["x"], [-1,28,28,1])

    #conv layer 1
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    #pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2],strides=2)

    #conv layer 2
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 64,
        kernel_size = [5,5],
        padding="same",
        activation=tf.nn.relu)

    #pool layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2],strides=2)
    
    pool2_flat = tf.reshape(pool2, [-1,7*7*64])

    #dense layer
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    #dropout
    dropout = tf.layers.dropout(
        inputs = dense, rate = 0.4, training=mode==tf.estimator.ModeKeys.TRAIN)

    #logits layer
    logits = tf.layers.dense(inputs=dropout, units=20)

    predictions = {
        "classes": tf.argmax(input=logits, axis = 1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeLeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)




if __name__ == "__main__":
    tf.app.run()
