#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import pickle

keyslist =  ['w1', 'W3', 'B1', 'B2', 'B3', 'W2']

parameters = {}

f =open('./model_saver/nn4_model.txt')

parameters = pickle.load(f)

print "parameters = ",(type(parameters),parameters.keys())
print "W1 = ",parameters['W1'].shape
print "B1 = ",parameters["B1"].shape
print "W2 = ",parameters['W2'].shape
print "B2 = ",parameters["B2"].shape
print "W3 = ",parameters['W3'].shape
print "B3 = ",parameters["B3"].shape