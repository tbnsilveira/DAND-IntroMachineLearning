#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
from sklearn import tree

### Defining the classifier
clf = tree.DecisionTreeClassifier(min_samples_split=40)

### Training the decision tree
clf = clf.fit(features_train, labels_train)

### Predicting
pred = clf.predict(features_test)

### Evaluating accuracy
acc_min_samples_split_40 = clf.score(features_test, labels_test)
print(acc_min_samples_split_40)


#########################################################
### How many features are in the data?
len(features_train)
len(features_train[1])
features_train.shape
# Data has 3785 attributes!









#########################################################


