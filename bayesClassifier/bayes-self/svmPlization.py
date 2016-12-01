#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import xlwt
import cv2
from bayesClassifier import BayesClassifier

dataset = np.load('pima-indians.npy')

columns = np.hsplit(dataset,9)
xsample = np.hstack(columns[0:8])
ysample = columns[8]
shape = xsample.shape
print "xsample = ",xsample.shape
print "ysample = ",ysample.shape
ysample = np.float32(ysample)

'''
book = xlwt.Workbook(encoding = 'utf-8')
sheet1 = book.add_sheet('Sheet1',cell_overwrite_ok = False)
columns = ['svm','svm1','svm2',]
for i in range(len(columns)):
    sheet1.write(0,i,columns[i])
data = []
for i in xrange(20):
    oneResult = []
    indexList = np.random.permutation(shape[0])
    # indexList = range(shape[0])
    classifier = BayesClassifier()
    classifier.sectionNumber = 12
    classifier.saveNeeded = False

    x_train = xsample[indexList[0:538]]
    y_train = ysample[indexList[0:538]]

    classifier.train(x_train,y_train)
    print "classifier train succefully ..."
    x_train1 = classifier.transform_pmodel(x_train)
    x_train2 = np.hstack([x_train,x_train1])
    print "x_train.shape = ",(x_train.shape,x_train1.shape,x_train2.shape)

    x_test = xsample[indexList[538:]]
    x_test1 = classifier.transform_pmodel(x_test)
    y_test = ysample[indexList[538:]]
    x_test2 = np.hstack([x_test,x_test1])
    print "x_test.shape = ",(x_test.shape,x_test1.shape,x_test2.shape)

    svm_params = dict( kernel_type = cv2.SVM_LINEAR,
	               svm_type = cv2.SVM_C_SVC,
                   C=3.67, gamma = cv2.SVM_GAMMA)
    svm = cv2.SVM()
    svm.train(x_train,y_train)
    ret = svm.predict_all(x_test)
    acurracy = (y_test == ret)
    svm_acurracy =np.mean(acurracy)
    oneResult.append(svm_acurracy)


    svm_params = dict( kernel_type = cv2.SVM_LINEAR,
	               svm_type = cv2.SVM_C_SVC,
                   C=3.67, gamma = cv2.SVM_GAMMA)
    svm1 = cv2.SVM()
    svm1.train(x_train1,y_train)
    ret1 = svm1.predict_all(x_test1)
    acurracy = (y_test == ret1)
    svm1_acurracy = np.mean(acurracy)
    oneResult.append(svm1_acurracy)

    svm_params = dict( kernel_type = cv2.SVM_LINEAR,
	               svm_type = cv2.SVM_C_SVC,
                   C=3.67, gamma = cv2.SVM_GAMMA)
    svm2 = cv2.SVM()
    svm2.train(x_train2,y_train)
    ret2 = svm2.predict_all(x_test2)
    acurracy = (y_test == ret2)
    svm2_acurracy = np.mean(acurracy)
    oneResult.append(svm2_acurracy)
    data.append(oneResult)
    print "oneResult = ",oneResult
    for j in xrange(len(oneResult)):
        sheet1.write(i+1,j,oneResult[j])
book.save('../result/svmPlization.xls')


data = np.array(data)
shape = data.shape

colors = ['r-','b-','g-','y-','m-','c-','k-']
shapes = ['ro','bo','go','yo','mo','co','ko']
from matplotlib import pyplot as plt
for j in range(shape[1]):
    plt.plot(data[:,j],colors[j])

plt.legend(columns)
for j in range(shape[1]):
    plt.plot(data[:,j],shapes[j])
plt.grid(True)
plt.xlabel('Experiment Times')
plt.ylabel('accuracy')
plt.title('Experiment Results')
plt.savefig('../images/svmPlization')
plt.show()
'''
'''
book = xlwt.Workbook(encoding = 'utf-8')
sheet1 = book.add_sheet('Sheet1',cell_overwrite_ok = False)
columns = ['MultinomialNB','MultinomialNB1','MultinomialNB2',]
for i in range(len(columns)):
    sheet1.write(0,i,columns[i])

data = []

for i in xrange(20):
    oneResult = []
    indexList = np.random.permutation(shape[0])
    # indexList = range(shape[0])
    classifier = BayesClassifier()
    classifier.sectionNumber = 12
    classifier.saveNeeded = False

    x_train = xsample[indexList[0:538]]
    y_train = ysample[indexList[0:538]]

    classifier.train(x_train,y_train)
    print "classifier train succefully ..."
    x_train1 = classifier.transform_pmodel(x_train)
    x_train2 = np.hstack([x_train,x_train1])
    print "x_train.shape = ",(x_train.shape,x_train1.shape,x_train2.shape)

    x_test = xsample[indexList[538:]]
    x_test1 = classifier.transform_pmodel(x_test)
    y_test = ysample[indexList[538:]]
    x_test2 = np.hstack([x_test,x_test1])
    print "x_test.shape = ",(x_test.shape,x_test1.shape,x_test2.shape)

    from sklearn.naive_bayes import MultinomialNB as BernoulliNB  
    clf = BernoulliNB()  
    clf.fit(x_train, y_train.ravel())  
    # BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)  
    predict =  clf.predict(x_test) 
    acurracy = (y_test == predict.ravel())
    acurracy =np.mean(acurracy)
    oneResult.append(acurracy)

 
    clf = BernoulliNB()  
    clf.fit(x_train1, y_train.ravel())  
    # BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)  
    predict =  clf.predict(x_test1) 
    acurracy = (y_test == predict.ravel())
    acurracy =np.mean(acurracy)
    oneResult.append(acurracy)


    clf = BernoulliNB()  
    clf.fit(x_train2, y_train.ravel())  
    # BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)  
    predict =  clf.predict(x_test2) 
    acurracy = (y_test == predict.ravel())
    acurracy =np.mean(acurracy)
    oneResult.append(acurracy)


    data.append(oneResult)
    print "oneResult = ",oneResult
    for j in xrange(len(oneResult)):
        sheet1.write(i+1,j,oneResult[j])
book.save('../result/MultinomialNBPlization.xls')


data = np.array(data)
shape = data.shape

colors = ['r-','b-','g-','y-','m-','c-','k-']
shapes = ['ro','bo','go','yo','mo','co','ko']
from matplotlib import pyplot as plt
for j in range(shape[1]):
    plt.plot(data[:,j],colors[j])

plt.legend(columns)
for j in range(shape[1]):
    plt.plot(data[:,j],shapes[j])
plt.grid(True)
plt.xlabel('Experiment Times')
plt.ylabel('accuracy')
plt.title('Experiment Results')
plt.savefig('../images/MultinomialNBPlization')
plt.show()
'''

'''

book = xlwt.Workbook(encoding = 'utf-8')
sheet1 = book.add_sheet('Sheet1',cell_overwrite_ok = False)
columns = ['AdaBoost','AdaBoost1','AdaBoost2',]
for i in range(len(columns)):
    sheet1.write(0,i,columns[i])

data = []

for k in xrange(20):
    oneResult = []
    indexList = np.random.permutation(shape[0])
    # indexList = range(shape[0])
    classifier = BayesClassifier()
    classifier.sectionNumber = 12
    classifier.saveNeeded = False

    x_train = xsample[indexList[0:538]]
    y_train = ysample[indexList[0:538]]

    classifier.train(x_train,y_train)
    print "classifier train succefully ..."
    x_train1 = classifier.transform_pmodel(x_train)
    x_train2 = np.hstack([x_train,x_train1])
    print "x_train.shape = ",(x_train.shape,x_train1.shape,x_train2.shape)

    x_test = xsample[indexList[538:]]
    x_test1 = classifier.transform_pmodel(x_test)
    y_test = ysample[indexList[538:]]
    x_test2 = np.hstack([x_test,x_test1])
    print "x_test.shape = ",(x_test.shape,x_test1.shape,x_test2.shape)

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             algorithm="SAMME",
                             n_estimators=200)
    bdt.fit(x_train,y_train.ravel())
    predict = bdt.predict(x_test)
    count = 0
    for i in xrange(y_test.shape[0]):
        if y_test[i,0] == predict[i]:
            count = count + 1
    acurracy = float(count)/y_test.shape[0]
    oneResult.append(acurracy)

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             algorithm="SAMME",
                             n_estimators=200)
    bdt.fit(x_train1,y_train.ravel())
    predict = bdt.predict(x_test1)
    count = 0
    for i in xrange(y_test.shape[0]):
        if y_test[i,0] == predict[i]:
            count = count + 1
    acurracy = float(count)/y_test.shape[0]
    oneResult.append(acurracy)


    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             algorithm="SAMME",
                             n_estimators=200)
    bdt.fit(x_train2,y_train.ravel())
    predict = bdt.predict(x_test2)
    count = 0
    for i in xrange(y_test.shape[0]):
        if y_test[i,0] == predict[i]:
            count = count + 1
    acurracy = float(count)/y_test.shape[0]
    oneResult.append(acurracy)


    data.append(oneResult)
    print "oneResult = ",oneResult

    for j in xrange(len(oneResult)):
        sheet1.write(k+1,j,oneResult[j])

book.save('../result/AdaBoostPlization.xls')


data = np.array(data)
shape = data.shape

colors = ['r-','b-','g-','y-','m-','c-','y-']
shapes = ['ro','bo','go','yo','mo','co','yo']
from matplotlib import pyplot as plt
for j in range(shape[1]):
    plt.plot(data[:,j],colors[j])

plt.legend(columns)
for j in range(shape[1]):
    plt.plot(data[:,j],shapes[j])
plt.grid(True)
plt.xlabel('Experiment Times')
plt.ylabel('accuracy')
plt.title('Experiment Results')
plt.savefig('../images/AdaBoostPlization')
plt.show()
'''

book = xlwt.Workbook(encoding = 'utf-8')
sheet1 = book.add_sheet('Sheet1',cell_overwrite_ok = False)
columns = ['knn','knn1','knn2',]
for i in range(len(columns)):
    sheet1.write(0,i,columns[i])

data = []

for i in xrange(20):
    oneResult = []
    indexList = np.random.permutation(shape[0])
    # indexList = range(shape[0])
    classifier = BayesClassifier()
    classifier.sectionNumber = 12
    classifier.saveNeeded = False

    x_train = xsample[indexList[0:538]]
    y_train = ysample[indexList[0:538]]

    classifier.train(x_train,y_train)
    print "classifier train succefully ..."
    x_train1 = classifier.transform_pmodel(x_train)
    x_train2 = np.hstack([x_train,x_train1])
    print "x_train.shape = ",(x_train.shape,x_train1.shape,x_train2.shape)

    x_test = xsample[indexList[538:]]
    x_test1 = classifier.transform_pmodel(x_test)
    y_test = ysample[indexList[538:]]
    x_test2 = np.hstack([x_test,x_test1])
    print "x_test.shape = ",(x_test.shape,x_test1.shape,x_test2.shape)

    knn = cv2.KNearest()
    knn.train(x_train,y_train)
    ret, results, neighbours ,dist = knn.find_nearest(x_test, 1)
    acurracy = (y_test == ret)
    acurracy = np.mean(acurracy)
    oneResult.append(acurracy)

    knn = cv2.KNearest()
    knn.train(x_train1,y_train)
    ret, results, neighbours ,dist = knn.find_nearest(x_test1, 1)
    acurracy = (y_test == ret)
    acurracy = np.mean(acurracy)
    oneResult.append(acurracy)

    knn = cv2.KNearest()
    knn.train(x_train2,y_train)
    ret, results, neighbours ,dist = knn.find_nearest(x_test2, 1)
    acurracy = (y_test == ret)
    acurracy = np.mean(acurracy)
    oneResult.append(acurracy)


    data.append(oneResult)
    print "oneResult = ",oneResult
    for j in xrange(len(oneResult)):
        sheet1.write(i+1,j,oneResult[j])
book.save('../result/knnPlization.xls')


data = np.array(data)
shape = data.shape

colors = ['r-','b-','g-','y-','m-','c-','k-']
shapes = ['ro','bo','go','yo','mo','co','ko']
from matplotlib import pyplot as plt
for j in range(shape[1]):
    plt.plot(data[:,j],colors[j])

plt.legend(columns)
for j in range(shape[1]):
    plt.plot(data[:,j],shapes[j])
plt.grid(True)
plt.xlabel('Experiment Times')
plt.ylabel('accuracy')
plt.title('Experiment Results')
plt.savefig('../images/knnPlization')
plt.show()