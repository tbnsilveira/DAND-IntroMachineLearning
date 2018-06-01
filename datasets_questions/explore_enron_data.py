#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


## How many data points (people) are in the dataset?
len(enron_data)

# For each person, how many features are available?
# First I read the keys of the "enron_data" and then run the command below:
len(enron_data['TAYLOR MITCHELL S'].keys())


# How many POIs are in the dataset?
## At this part, I will opt to use pandas:
import pandas as pd

enron = pd.DataFrame.from_dict(enron_data, orient='index')
enron[enron['poi']==True]

## It returns there is 18 rows, i.e. 18 POI.


### The next questions are concerning the dataset:
features = enron.columns  #Just to check
