#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("/home/tbnsilveira/workplace/tef_udacity/7. Intro to Machine Learning/DAND-IntroMachineLearning/tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
clf = SVC(kernel='linear')

## Treinando o modelo:
t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

## Testando o modelo:
t1 = time()
clf.predict(features_test)
print("training time:", round(time()-t1, 3), "s")

## Avaliando o resultado:
clf.score(features_test,labels_test)
#########################################################

### Reducing the training size:
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 

## Treinando o modelo:
t2 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

## Testando o modelo:
t3 = time()
clf.predict(features_test)
print("training time:", round(time()-t1, 3), "s")

## Avaliando o resultado:
clf.score(features_test,labels_test)


#########################################################
## Now training with an 'rbf' kernel (and still 1% of data)
#########################################################
c_values = [1.0,10.0,100.0,1000.0,10000.0]

for c in c_values:
    print(c)
    
    clf = SVC(kernel='rbf',C=c)

    ## Treinando o modelo:
    t4 = time()
    clf.fit(features_train, labels_train)
    print("training time:", round(time()-t0, 3), "s")
    
    ## Testando o modelo:
    t5 = time()
    clf.predict(features_test)
    print("training time:", round(time()-t1, 3), "s")
    
    ## Avaliando o resultado:
    score = clf.score(features_test,labels_test)
    print('SVM score for c=',c,': ',score)
    
#########################################################
## Now training with an 'rbf' kernel, C=10.0000 and for the full dataset
#########################################################
features_train, features_test, labels_train, labels_test = preprocess()

clf = SVC(kernel='rbf',C=10000.0)

## Treinando o modelo:
t6 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

## Testando o modelo:
t7 = time()
pred = clf.predict(features_test)
print("training time:", round(time()-t1, 3), "s")

## Avaliando o resultado:
score = clf.score(features_test,labels_test)
print('SVM score for c=',c,': ',score)



#########################################################
## Counting how many predictions to each user:
#########################################################
import pandas as pd
teste = pd.DataFrame(pred)
teste = pd.DataFrame(pred,columns=['Predict'])
teste[teste['Predict']==0]
teste[teste['Predict']==1]




