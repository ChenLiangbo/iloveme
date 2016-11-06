import os 
import numpy as np
import cv2


outdir = './finalfile/'

x_train = np.float32(np.load(outdir + 'x_train.npy'))
y_train = np.float32(np.load(outdir + 'y_train.npy'))


shape = y_train.shape
y = np.zeros(shape)
for i in xrange(shape[0] - 1):
    if y_train[i+1,0] < y_train[i,0]:
        y[i,0] = 0
    else:
        y[i,0] = 1
y_train = np.float32(y)

x_test = np.float32(np.load(outdir + 'x_test.npy'))
y_test = np.float32(np.load(outdir + 'y_test.npy'))
shape = y_test.shape
y = np.zeros(shape)
for i in xrange(shape[0] - 1):
    if y_test[i+1,0] < y_test[i,0]:
        y[i,0] = 0
    else:
        y[i,0] = 1
y_test = np.float32(y)

length = x_train.shape[1]


knn = cv2.KNearest()
knn.train(x_train,y_train)
ret, results, neighbours ,dist = knn.find_nearest(x_test,1)

# results = results.ravel()
acurracy = (y_test == results)

print "knn acurracy = ",np.mean(acurracy)




svm_params = dict( kernel_type = cv2.SVM_LINEAR,
	               svm_type = cv2.SVM_C_SVC,
                   C=1, gamma=0.5)

svm = cv2.SVM()
svm.train(x_train,y_train,params = svm_params)
# svm.save('svm_mnist_model.dat')
ret = svm.predict_all(x_test)
results =ret
acurracy = (y_test == results)

print "svm acurracy = ",np.mean(acurracy)




# random trees
RTtree = cv2.RTrees()

rtree_params = dict(depth = 32)
var_type = np.array([cv2.CV_VAR_NUMERICAL]*length + [cv2.CV_VAR_CATEGORICAL],dtype = np.uint8)


RTtree.train(x_train,cv2.CV_ROW_SAMPLE,y_train,
    	varType = var_type,
    	params = rtree_params)

results = np.zeros(y_test.shape,dtype = y_test.dtype)
for i in xrange(y_test.shape[0]):
    results[i,0] = RTtree.predict(x_test[i,:])


acurracy = (y_test == results)

print "RTtree acurracy = ",np.mean(acurracy)




boost_params = dict(max_depth = 1)
var_type = np.array([cv2.CV_VAR_NUMERICAL]*length + [cv2.CV_VAR_CATEGORICAL],dtype = np.uint8)
Boost = cv2.Boost()
Boost.train(x_train,cv2.CV_ROW_SAMPLE,y_train,varType = var_type,params = boost_params)

results = np.zeros(y_test.shape,dtype = y_test.dtype)

for i in xrange(y_test.shape[0]):
    results[i,0] = Boost.predict(x_test[i,:],returnSum = True)

print "results = ",results[0:10]

for i in xrange(y_test.shape[0]):
    if results[i,0] >0:
        results[i,0] = 1
    else:
        results[i,0] = 0


acurracy = (y_test == results)

print "Boost acurracy = ",np.mean(acurracy)





