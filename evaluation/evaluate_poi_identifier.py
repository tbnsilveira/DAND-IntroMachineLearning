#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


#%%## it's all yours from here forward!  
from sklearn import tree
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### Defining the classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

### Predicting
pred = clf.predict(features_test)

### Evaluating accuracy
acc = clf.score(features_test, labels_test)
print(acc)

#%% Evaluating (Class 31. Number of TP)
## Look at the predictions of your model and compare them to the true test labels. Do you get any true positives? 
## (In this case, we define a true positive as a case where both the actual label and the predicted label are 1)
print('Pred \t Real')
for i in range(len(pred)):
    print('{}\t{}'.format(pred[i],labels_test[i]))











