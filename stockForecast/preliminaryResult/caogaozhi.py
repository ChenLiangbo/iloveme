import os 
import numpy as np
import cv2


outdir = './finalfile/'
x_sample = np.load(outdir + 'x_train_normalized.npy')
y_sample = np.load(outdir + 'y_train_normalized.npy')


train_start = 600
train_end = 1150

test_start = 1000
test_end = 1200

x_train = x_sample[train_start:train_end,:]
x_test = x_sample[test_start:test_end,:]

np.save(outdir + 'x_train',x_train)
np.save(outdir + 'x_test',x_test)