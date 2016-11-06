import cv2
import numpy as np
import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#knn
# itertimes = 50
# knn = cv2.KNearest()
# for i in xrange(itertimes):
#     sample = mnist.train.next_batch(50)
#     x_train,y_train = sample[0],sample[1]

#     yshape = y_train.shape
#     y = np.zeros((y_train.shape[0],1),dtype = np.float32)
#     for row in range(yshape[0]):
#         index = np.argwhere(y_train[row] == 1)
#         index = index[0,0]
#         y[row,0] = index

#     knn.train(x_train,y)
    
#     test_batch = mnist.train.next_batch(50)
#     x_test,y_test = test_batch[0],test_batch[1]
#     y = np.zeros((y_test.shape[0],1),dtype = np.float32)
#     for row in range(yshape[0]):
#         index = np.argwhere(y_test[row] == 1)
#         index = index[0,0]
#         y[row,0] = index

#     ret, results, neighbours ,dist = knn.find_nearest(x_test, 3)

#     acurracy = (y == results)

#     print "step = %d,acurracy = %f " % (i,np.mean(acurracy))

# print "finished ..."

sample = mnist.train.next_batch(5000)
x_train,y_train = sample[0],sample[1]

ann = cv2.ANN_MLP()

layer_sizes = np.int32([784,1000,30,10])  
ann.create(layer_sizes)  

params = dict( term_crit = (cv2.TERM_CRITERIA_COUNT, 1000, 0.001),  
                   train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,  
                   bp_dw_scale = 0.001,  
                   bp_moment_scale = 0.0 ) 


ann.train(x_train,y_train,None,params = params)

test = mnist.train.next_batch(1000)
x_test,y_test = test[0],test[1]

ret, results = ann.predict(x_test)

print "ret = ",ret[0:10]
#print "result = ",results[0:10]

ret = results
#print "ret.shape = ",ret.shape
for i in range(ret.shape[0]):
    if ret[i,0] > 0.5:
        ret[i,0] = 1
    else:
        ret[i,0] = 0

acurracy = (y_train == ret)
# print "results = ",results[0:10,:]

print "acurracy = %f " % (np.mean(acurracy),)