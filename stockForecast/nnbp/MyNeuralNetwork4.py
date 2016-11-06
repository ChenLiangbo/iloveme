#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import pickle

class MyNeuralNetwork4(object):

    def __init__(self,):
        super(MyNeuralNetwork4,self).__init__()

        self.X = tf.placeholder("float", [None, nn_input])
        self.Y = tf.placeholder("float", [None, nn_output])
        self.inputNumber  = 5
        self.layerOne     = 1200
        self.layerTwo     = 30
        self.outputNumber = 1
        self.learnRate    = 0.01    

    def init_weight(self,shape,name = None):
        return tf.Variable(tf.random_normal(shape, stddev=0.01),name = name)


    def init_bias(self,shape,name = None):
        init = tf.zeros(shape)
        return tf.Variable(init, name=name)


    def model(self,X,W,B):
        m = tf.matmul(X, W) + B
        # RELU for instead sigmoid, Sigmoid only for Final
        L = tf.nn.tanh(m)
        return L

    def neural_network(self,):

        W1 = init_weight([self.inputNumber, self.layerOne], 'W1')
        B1 = init_bias([self.layerOne], 'B1')

        W2 = init_weight([self.layerOne, self.layerTwo], 'W2')
        B2 = init_bias([self.layerTwo], 'B2')


        W3 = init_weight([self.layerTwo, self.outputNumber], 'W3')
        B3 = init_bias([self.outputNumber], 'B3')

        L2 = model(self.X,  W1, B1)
        L3 = model(L2, W2, B2)
        y_out = tf.nn.relu(tf.matmul(L3, W3) + B3)

        return y_out

    def initSession(self,):
        sess = tf.Session()
        init = tf.initialize_all_variables()

    def save_model(self,):
        pass

    def reload_model(self,):
        pass