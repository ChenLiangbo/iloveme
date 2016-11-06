#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),name="weights")
print "weights type = ",type(weights)
print "dir weights = ",dir(weights)
init_op = tf.initialize_all_variables()

c = tf.constant(4.0)

with tf.Session() as sess:
# Run the init operation.
    sess.run(init_op)
    print "weights = ",weights.eval()
    print "weights type = ",type(weights)
    print "dir weights = ",dir(weights)