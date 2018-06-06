#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


#%%## read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL',0)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

#%% Visualizing the dataset
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


#%% Identifying the outlier point:
## The easiest way to find out the outlier was using pandas:
import pandas as pd

dataPD = pd.DataFrame(data)
dataPD[0].max()  ##Looking for this value at the pdf table.

## It was identified as the 'TOTAL', which is in turn removed by line 12.

#%% Identifying two more outliers:
## "Two people made bonuses of at least 5 million dollars, and a salary of over 1 million dollars"

dataset = pd.DataFrame.from_dict(data_dict,orient='index')

# Since the numerical data was imported as string, we need first to clen the data:
removeBonusIx = dataset[dataset['bonus']=='NaN'].index
dataset.drop(removeBonusIx, axis=0, inplace=True)

removeSalaryIx = dataset[dataset['salary']=='NaN'].index
dataset.drop(removeSalaryIx, axis=0, inplace=True)

# Converting now to integer:
dataset['bonus'] = dataset.bonus.apply(int)
dataset['salary'] = dataset.salary.apply(int)

## Finding the POI:
dataset[(dataset['salary'] > 1000000) & (dataset['bonus'] > 5000000)].index
