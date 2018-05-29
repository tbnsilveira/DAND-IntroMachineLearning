#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
#ys.path.append("../tools/")
sys.path.append("/home/tbnsilveira/workplace/tef_udacity/7. Intro to Machine Learning/DAND-IntroMachineLearning/tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

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
