#!/usr/bin/python

import sys
import os
import pandas as pd
import pickle
import pprint as pp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

plt.style.use('ggplot')

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
				 'salary',
				 'bonus', 
				 'deferral_payments',
				 'director_fees',
				 'exercised_stock_options',
				 'expenses',
				 'from_messages',
				 'from_poi_to_this_person',
				 'from_this_person_to_poi',
				 'loan_advances',
				 'long_term_incentive',
				 'restricted_stock',
				 'restricted_stock_deferred',
				 'to_messages',
				 'total_payments',
				 'total_stock_value'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print len(features_list)
### Task 2: Remove outliers

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
data = featureFormat(data_dict, features_list)
df = pd.DataFrame(data, columns = features_list)

# we will store our feature's title and their respective outliers instances in
# dictionary

dictionary = {}

# Making our standard deviation driscrimination system:

def greater_than_xstds(integer, std , mean, multiplier = 3):

	difference = abs(integer - mean)
	x = difference/std
	if x >= multiplier:
		return True
	else:
		return False

# Here we will create a function which will use our greater_than_xstds function
# in order to find our outliers and log their respective information.

def check_for_outliers(df, feature, xstd = 3):

	std = df[feature].std()
	mean = df[feature].mean()
	dictionary[feature] = []
	position = 0 

	for value in df[feature]:
		if greater_than_xstds(value, std, mean, multiplier = xstd):
			index_v = dictionary[feature].append((position,
												  value,
												  mean,
												  std))
		else:
			pass
		position += 1
	return dictionary

outliers = 0

# Here we will use our check_for_outliers function to loop through every feature
# in search of outliers

for feature in features_list:
	a = check_for_outliers(df, feature, xstd = 2.5)
	if a:
		outliers = a

# pp.pprint(outliers)

# let's make a list of all outlier instance by index number, and then a set of 
# these row indexes to eliminat the rown from our dataframe as a whole

outliers_index_numbers = []

for feature in outliers:
	for tupl in outliers[feature]:
		outliers_index_numbers.append(tupl[0])

outliers_index_numbers = set(outliers_index_numbers)

# To print, uncomment the enclosed lines:
################################################################################
# print
# print "Rows with outliers: ", len(outliers_index_numbers)
# print
# print "Row indexes: "
# pp.pprint(outliers_index_numbers)
################################################################################ 

# Now let's eliminate these

# Here's is the original size (Uncomment):
# print
# print "Original size: ",df.shape
# print

# # Dropping

print 

df.drop(outliers_index_numbers, inplace = True)

# Size after dropping (Uncomment):
# print
# print "After dropping outlier rows: ",df.shape
# print



################################################################################
### Also, I am pickling out this dictionary to perform further independent analysis.
### link to all code and data is provided in my Notes.py file:

# with open("Outliers_dictionary.pkl", "w") as f:
# 	pickle.dump(outliers, f)

################################################################################

### Task 3: Create new feature(s)

print df.poi[ df.poi == 0.0]

# ### Store to my_dataset for easy export below.
# my_dataset = data_dict

# ### Extract features and labels from dataset for local testing
# data = featureFormat(my_dataset, features_list, sort_keys = True)
# labels, features = targetFeatureSplit(data)

# ### Task 4: Try a varity of classifiers
# ### Please name your classifier clf for easy export below.
# ### Note that if you want to do PCA or other multi-stage operations,
# ### you'll need to use Pipelines. For more info:
# ### http://scikit-learn.org/stable/modules/pipeline.html

# # Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

# ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
# ### using our testing script. Check the tester.py script in the final project
# ### folder for details on the evaluation method, especially the test_classifier
# ### function. Because of the small size of the dataset, the script uses
# ### stratified shuffle split cross validation. For more info: 
# ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# # Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

# ### Task 6: Dump your classifier, dataset, and features_list so anyone can
# ### check your results. You do not need to change anything below, but make sure
# ### that the version of poi_id.py that you submit can be run on its own and
# ### generates the necessary .pkl files for validating your results.

# dump_classifier_and_data(clf, my_dataset, features_list)