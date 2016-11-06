#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
from tensorflow.python.platform import gfile


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summaries_dir', './save_graph_logs', 'Summaries directory')

data = np.arange(10,dtype=np.int32)

with tf.Session() as sess:
  print("# build graph and run")
  input1= tf.placeholder(tf.int32, [10], name="input")
  output1= tf.add(input1, tf.constant(100,dtype=tf.int32), name="output") #  data depends on the input data
  saved_result= tf.Variable(data, name="saved_result")
  do_save=tf.assign(saved_result,output1)
  tf.initialize_all_variables()
  os.system("rm -rf ./save_graph_logs")
  merged = tf.merge_all_summaries()
  train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir,sess.graph)
  os.system("rm -rf ./load")
  tf.train.write_graph(sess.graph_def, "./load", "test.pb", False) #proto
  # now set the data:
  result,_=sess.run([output1,do_save], {input1: data}) # calculate output1 and assign to 'saved_result'
  saver = tf.train.Saver(tf.all_variables())
  saver.save(sess,"checkpoint.data")