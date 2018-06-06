#!/usr/bin/python

import random
import numpy
import matplotlib.pyplot as plt
import pickle

from sklearn import linear_model


#%% There was an error importing the file. Declaring the function here:
#from outlier_cleaner.py import outlierCleaner
def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ## Calculating the error:
    errors = net_worths - predictions
    ## Creating a dictionary for error position on the lists:
    errorPos = {}
    for ix, error in enumerate(errors):
        errorPos[str(error)] = ix
    
    errorsSort = errors[:] 
    errorsSort.sort(axis=0)  #Sorting the errors values
    errorsSort = errors[9:]  #Selecting the 81 higher elements
    
    for error in errorsSort:
        ix = errorPos[str(error)]
        sample = (ages[ix], net_worths[ix], errors[ix])
        cleaned_data.append(sample)
        
    return cleaned_data

#%% load up some practice data with outliers in it
ages = pickle.load( open("../outliers/practice_outliers_ages.pkl", "r") )
net_worths = pickle.load( open("../outliers/practice_outliers_net_worths.pkl", "r") )

### ages and net_worths need to be reshaped into 2D numpy arrays
### second argument of reshape command is a tuple of integers: (n_rows, n_columns)
### by convention, n_rows is the number of data points
### and n_columns is the number of features
ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

from sklearn.cross_validation import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

## Visualizing the points:
plt.scatter(ages, net_worths)
plt.show()

#%% fill in a regression here!  Name the regression object reg so that
### the plotting code below works, and you can see what your regression looks like
reg = linear_model.LinearRegression()
reg.fit (ages_train, net_worths_train)

print('Regression score on the testing data: ',reg.score(ages_test,net_worths_test))
print('The slope of the regression is ',reg.coef_)

#%% Visualizing the trained data:
try:
    plt.plot(ages, reg.predict(ages), color="blue")
except NameError:
    pass
plt.scatter(ages, net_worths)
plt.show()


#%%identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    cleaned_data = outlierCleaner( predictions, ages_train, net_worths_train )
except NameError:
    print "your regression object doesn't exist, or isn't name reg"
    print "can't make predictions to use in identifying outliers"


#%% only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

    ### refit your cleaned data!
    try:
        reg.fit(ages, net_worths)
        plt.plot(ages, reg.predict(ages), color="blue")
    except NameError:
        print "you don't seem to have regression imported/created,"
        print "   or else your regression object isn't named reg"
        print "   either way, only draw the scatter plot of the cleaned data"
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()


else:
    print "outlierCleaner() is returning an empty list, no refitting to be done"

