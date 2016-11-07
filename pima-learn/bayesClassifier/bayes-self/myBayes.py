#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
from bayesClassifier import BayesClassifier

dataset = np.load('pima-indians.npy')

columns = np.hsplit(dataset,9)
xsample = np.hstack(columns[0:8])
ysample = columns[8]
shape = xsample.shape
print "xsample = ",xsample.shape
print "ysample = ",ysample.shape

# indexList = np.random.permutation(shape[0])
indexList = range(shape[0])

x_train = xsample[indexList[0:538]]
y_train = ysample[indexList[0:538]]
print "x_train.shape = ",x_train.shape
print "y_train.shape = ",y_train.shape

x_test = xsample[indexList[538:]]
y_test = ysample[indexList[538:]]
print "x_test.shape = ",x_test.shape
print "y_test.shape = ",y_test.shape


classifier = BayesClassifier()
classifier.saveNeeded = False
classifier.sectionNumber = 32
classifier.train(x_train,y_train)
print "classifier train succefully ..."
y_predict = classifier.predict(x_test)

# print "y_predict = ",y_predict
print "y_predict = ",y_predict.shape
accuracy = (y_test == y_predict)

print "BayesClassifier accuracy = ",np.mean(accuracy)
print "-"*100



from sklearn.naive_bayes import BernoulliNB  
clf = BernoulliNB()  
clf.fit(x_train, y_train.ravel())  
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)  
predict =  clf.predict(x_test) 

acurracy = (y_test == predict.ravel())

print "BernoulliNB acurracy = %f" % (np.mean(acurracy)) 


import cv2
svm_params = dict( kernel_type = cv2.SVM_LINEAR,
	               svm_type = cv2.SVM_C_SVC,
                   C=3.67, gamma = cv2.SVM_GAMMA)

svm = cv2.SVM()
svm.train(x_train,y_train)
ret = svm.predict_all(x_test)
acurracy = (y_test == ret)
print "svm acurracy = ",np.mean(acurracy)

knn = cv2.KNearest()
knn.train(x_train,y_train)
ret, results, neighbours ,dist = knn.find_nearest(x_test, 1)
acurracy = (y_test == ret)
print "knn acurracy = ",np.mean(acurracy)

from sklearn.naive_bayes import MultinomialNB  
clf = MultinomialNB().fit(np.abs(x_train), y_train.ravel())  
predict =  clf.predict(np.abs(x_test)) 
acurracy = (y_test == predict.ravel())
print "MultinomialNB acurracy = %f" % (np.mean(acurracy))


from sklearn.naive_bayes import GaussianNB  
clf = GaussianNB().fit(x_train, y_train.ravel())  
predict =  clf.predict(x_test) 
acurracy = (y_test == predict.ravel())
print "GaussianNB acurracy = %f" % (np.mean(acurracy))


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
print "AdaBoostClassifier acurracy = ",float(count)/y_test.shape[0]
print "="*80

'''

result = []
index = []
start = 5
for i in range(start,50):
    classifier = BayesClassifier()
    classifier.saveNeeded = False
    classifier.sectionNumber = i

    try:
        classifier.train(x_train,y_train)
        y_predict = classifier.predict(x_test)
        accuracy = (y_test == y_predict)
        result.append(np.mean(accuracy))
    except Exception,ex:
        print "[Exception] ",str(ex)
        result.append(0.5) 
    index.append(i)   
    print "i = ",i
    print "-"*100
maxValue = np.max(result)
maxIndex = result.index(maxValue)
print "maxValue = %f,sectionNumber = %d" % (maxValue,index[maxIndex])
from matplotlib import pyplot as plt
x = range(start,50)
plt.plot(x,result,'ro')
plt.plot(x,result,'r-')
plt.grid(True)
plt.legend(['sectionNumber',])
plt.xlabel('Value of sectionNumber')
plt.ylabel('acurracy')
plt.title('Distribution of Acurracy With sectionNumber')
plt.savefig('../images/sectionNumber-accuracy')
plt.show()
'''

import xlwt

book = xlwt.Workbook(encoding = 'utf-8')
sheet1 = book.add_sheet('Sheet1',cell_overwrite_ok = False)
columns = ["myBayes","BernoulliNB","svm","knn","MultinomialNB","GaussianNB","AdaBoost"]
for i in range(7):
    sheet1.write(0,i,columns[i])

data = []
for ci in range(20):
    valueList = []
    indexList = np.random.permutation(shape[0])
    # indexList = range(shape[0])
    
    x_train = xsample[indexList[0:538]]
    y_train = ysample[indexList[0:538]]
    print "x_train.shape = ",x_train.shape
    print "y_train.shape = ",y_train.shape

    x_test = xsample[indexList[538:]]
    y_test = ysample[indexList[538:]]
    print "x_test.shape = ",x_test.shape
    print "y_test.shape = ",y_test.shape
    
    
    classifier = BayesClassifier()
    classifier.saveNeeded = False
    classifier.sectionNumber = 30
    classifier.train(x_train,y_train)
    y_predict = classifier.predict(x_test)   
    print "y_predict = ",y_predict.shape
    accuracy = (y_test == y_predict)
    a1 = np.mean(accuracy)
    print "BayesClassifier accuracy = ",a1
    valueList.append(a1)

    from sklearn.naive_bayes import BernoulliNB  
    clf = BernoulliNB()  
    clf.fit(x_train, y_train.ravel())  
    BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)  
    predict =  clf.predict(x_test) 
    
    acurracy = (y_test == predict.ravel())
    a2 = (np.mean(acurracy))
    print "BernoulliNB acurracy = %f" %  (a2,)
    valueList.append(a2)
    
    
    import cv2
    svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                       svm_type = cv2.SVM_C_SVC,
                   C=3.67, gamma = cv2.SVM_GAMMA)

    svm = cv2.SVM()
    svm.train(x_train,y_train)
    ret = svm.predict_all(x_test)
    acurracy = (y_test == ret)
    a3 = np.mean(acurracy)
    print "svm acurracy = ",a3
    valueList.append(a3)

    knn = cv2.KNearest()
    knn.train(x_train,y_train)
    ret, results, neighbours ,dist = knn.find_nearest(x_test, 1)
    acurracy = (y_test == ret)
    a4 = np.mean(acurracy)
    print "knn acurracy = ",a4
    valueList.append(a4)
    
    from sklearn.naive_bayes import MultinomialNB  
    clf = MultinomialNB().fit(np.abs(x_train), y_train.ravel())  
    predict =  clf.predict(np.abs(x_test)) 
    acurracy = (y_test == predict.ravel())
    a5 = np.mean(acurracy)
    print "MultinomialNB acurracy = %f" % (a5,)
    valueList.append(a5)
    
    
    from sklearn.naive_bayes import GaussianNB  
    clf = GaussianNB().fit(x_train, y_train.ravel())  
    predict =  clf.predict(x_test) 
    acurracy = (y_test == predict.ravel())
    a6 = np.mean(acurracy)
    print "GaussianNB acurracy = %f" % (a6,)
    valueList.append(a6)

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
    a7 = float(count)/y_test.shape[0]
    print "AdaBoostClassifier acurracy = ",a7
    valueList.append(a7)
    
    print "valueList = ",len(valueList)
    print "="*80
    for j in range(7):
        sheet1.write(ci+1,j,valueList[j])

    data.append(valueList)
book.save('../result/pima-indians.xls')

data = np.array(data)
shape = data.shape
print "shape = ",shape

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
plt.savefig('../images/experiment1')
plt.show()

