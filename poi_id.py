from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from numpy import inf
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import tree 
from sklearn import svm
from sklearn import preprocessing

import numpy as np
import os
import pandas as pd
import pickle
import pprint as pp
import string
import sys


dp = sys.path[0]

################################################################################

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


print 
print "Task 1"
print


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
with open("final_project_dataset.pkl", "r") as pkl:
    dataset = pickle.load(pkl)

names = [i for i in dataset]
names.sort()



# ##  Making a pandas dataframe.

# ###  Dictionary re-write.

# Here we rewrite the dataset nested dictionaries into a list of dictionaries,
# each of which now has a new key-value pair ('name' : EMPLOYEE NAME). This is 
# to easily turn this new list into a pandas dataframe.

def dict_rewrite(dataset):
  
    l = []
    for name in dataset:
        d = {}
        d['name'] = name
        for heading in dataset[name]:
            d[heading] = dataset[name][heading]
        l.append(d)
    return l

list_of_dicts = dict_rewrite(dataset) 

# pp.pprint(list_of_dicts) # Uncomment this line to see the list of dictionaries



# ### From list of dictionaries to pandas-dataframe.

# let's turn list_of_dicts into a pandas df

df = pd.DataFrame(list_of_dicts)

# Now, let's reassign our name column as our index column

df.set_index('name', inplace = True)

# We use the argument inplace to reassign the value of our original variable 
# name, df, to our new, modified dataframe.

df.sort_index(inplace = True)

# Looking through the list I noticed a row denominated as "TOTAL"
# this is clearly a mistake, so let's just remove it:

df.drop('TOTAL', inplace = True)



# ### Making records 

email_address = df.email_address.copy

df.drop('email_address', axis = 1, inplace =True)

# df.columns # Uncomment this line to see the dataframes columns names

features_list = [i for i in df.columns]

for feature in features_list:
    df[feature].replace(to_replace = "NaN", value = 0, inplace= True)

print 
print "completed"
print

### Task 2: Remove outliers

print 
print "Task 2"
print

# We will store our feature's title and their respective outliers instances in
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

def check_for_outliers(df, feature, xstd = 3.0):

    std = df[feature].std()
    mean = df[feature].mean()
    dictionary[feature] = []
    position = 0 

    for value in df[feature]:
        if greater_than_xstds(value, std, mean, multiplier = xstd):
            dictionary[feature].append((position,
                                        value,
                                        ))
        else:
            pass
        position += 1
    return dictionary

outliers = 0

# Here we will use our check_for_outliers function to loop through every feature
# in search of outliers.

def checking_features(features_list, df):

	outliers = 0

	for feature in features_list:
	    a = check_for_outliers(df, feature, xstd = 1.8)
	    if a:
	        outliers = a

	return outliers

outliers = checking_features(features_list, df)

# pp.pprint(outliers) # Uncomment this line to see the list of outliers

# let's make a list of all outlier instance by index number, and then a set of 
# these row indexes to remove the row from our dataframe as a whole into a new
# dataframe of its own

outliers_index_numbers = []

for feature in outliers:
    for tupl in outliers[feature]:
        outliers_index_numbers.append(tupl[0])

outliers_index_numbers = set(outliers_index_numbers)

outliers_index_numbers = [i for i in outliers_index_numbers]

# print
# print "Rows with outliers: ", len(outliers_index_numbers)
# print
# print "Row indexes:"
# print
# print(outliers_index_numbers)

outliers_df = df.iloc[outliers_index_numbers]

print "Outlier names"
print
print([i for i in outliers_df[outliers_df["poi"] == 0].index.values])
print


df_without_outliers = df.drop(outliers_df[outliers_df["poi"] == 0].index.values,
							  axis = 0)

print 
print "completed"
print

### Task 3: Create new feature(s)

print 
print "Task 3"
print

# pca will now be applied in our to find those features of greatest importances
# from our original final_project_dataset.pkl file

def pca(df_without_outliers):
    
    df1 = df_without_outliers.copy()    
    
    labels = df1.poi.copy()
    features = df1.drop("poi", axis = 1)

    features_names = np.array(features.columns.tolist())
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
    													features,
    												labels, test_size = 0.20)

    selector = SelectPercentile(f_classif, percentile = 20)
    selector.fit(X_train, y_train)
    
    important_features = selector.get_support(indices=False)

    scores = selector.scores_
    
    scores = scores[important_features].tolist() 

    pca_features = features_names[important_features].tolist()

    scores_report =\
	{pca_features[i]:scores[i] for i in range(len(pca_features))}
    
    if 'poi' not in pca_features:
        pca_features.append('poi')

    return pca_features, scores_report



pca_features, scores_report = pca(df_without_outliers)

print
print "Financial features of interest scores report: "
print
pp.pprint(scores_report)
print
print 

df_without_outliers = df_without_outliers[pca_features]


# ## Making last names abbreviation

# this is the rewrite of our suspects' names to match them to their respective
# email folders.


def find_last_name(df_column):

    for name in df_column:
        # Split name list
        snl = name.split(' ')
        df_without_outliers.loc[name, 'last_name'] = ("{}-{}").format(
        													snl[0].lower(),
        													snl[1][0].lower())
        
find_last_name(df_without_outliers.index)


employee_last_names = [i for i in df_without_outliers['last_name'].values]
employee_file_names = os.listdir("maildir/")


# ## Finding filenames in maildir

# Now we match these name abbreviations

def find_matching_filenames(df_without_outliers,
							employee_last_names,
							employee_file_names):
    
    employees_w_email_dir = []
    employees_email_dirs =[]

    for last_name in employee_last_names:
        for file_name in employee_file_names:
            if last_name == file_name:
                employees_email_dirs.append(file_name)
                
    for abbreviation in employees_email_dirs:
        employees_w_email_dir.append(df_without_outliers.index\
        			  [df_without_outliers.last_name == abbreviation].values[0])
    
    return employees_w_email_dir, employees_email_dirs

employees_w_email_dir, employees_email_dirs = find_matching_filenames(
															df_without_outliers,
															employee_last_names,
															employee_file_names)


# ## Parsing emails for new features

# with the help of nltk we will define a function to parse the respective emails
# of each employee in order to create new features. 

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """

    sw = stopwords.words("english")
    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""),
        string.punctuation)

        ### project part 2: comment out the line below
        # words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        
        stemmer = SnowballStemmer("english")
        list_of_words = text_string.split(' ')

        ########################################################################
        # Added to remove stopwords:

        for word in list_of_words:
            for sword in sw:
                if str(word) == sword:
                    try:
                        del list_of_words[list_of_words.index(word)]

                    except:
                        pass
        ########################################################################

        list_of_words_stemmed = \
         [stemmer.stem(word.strip()) for word in list_of_words]
        words = ' '.join(list_of_words_stemmed)
        words = str(words)

    return words


def get_email_root_words(df_without_outliers, employees_email_dirs):

    if ("list_of_vocabs.pkl" in os.listdir(os.getcwd()))\
     and ("emp_name_abs.pkl" in os.listdir(os.getcwd())):

        with open('list_of_vocabs.pkl', 'r') as f:
            list_of_vocabs = pickle.load(f)

        with open('emp_name_abs.pkl', 'r') as f:
            emp_name_abs = pickle.load(f) 

        return list_of_vocabs, emp_name_abs

    else:

        emp_name_abs = []
        list_of_vocabs = []
        
        # Let's switch to our tools folder to get our parsing function
        
        #Now, let's switch to our maildir folder to get our emails
        employee_names = [i for i in df_without_outliers.last_name.values]
        index_number = 0

        print
        print "Obtaining respective employees emails' root words: "
        print
        
        for employee_name in employees_email_dirs:
            
            string_of_words = 0
            os.chdir(dp+"\maildir")

            try:

                string_of_words = []
                os.chdir(dp+"\maildir\{}".format(employee_name))
                print os.getcwd()
                files = [i for i in os.listdir(os.getcwd())]

                for f in files:
                    
                    os.chdir(dp+"\maildir\{}\{}".format(employee_name,f))
                    emails = [i for i in os.listdir(os.getcwd())]
                    
                    for email in emails:
                    
                        try:

                            f = open(email, "r")
                            string_of_words.append(parseOutText(f)) 

                        except:
                            pass

                string_of_words = " ".join(string_of_words)

                list_of_vocabs.append(string_of_words)
                emp_name_abs.append([employee_name,
                df_without_outliers.poi[df_without_outliers.last_name == \
                employee_name][0]])

            except:    
                pass
            
        df_without_outliers = df_without_outliers.drop('last_name', axis = 1)

        print len(emp_name_abs)
        print len(list_of_vocabs)

        os.chdir(dp)

        with open('emp_name_abs.pkl','w') as f:
            pickle.dump(emp_name_abs,f) 

        with open('list_of_vocabs.pkl','w') as f:
            pickle.dump(list_of_vocabs,f)

        
        
        return list_of_vocabs, emp_name_abs

# ### Calling our function

# All of the emails have already been parsed and pickled, alongside to their
# respective employee "poi" status, in the necessary order respectively in the
# interest of saving time. you can skip ahead to the "Loading email data"
# section below to load these.



list_of_vocabs, emp_name_abs = get_email_root_words(df_without_outliers,
														employees_email_dirs)

# let's dump these lists



# since emp_name_abs is filled with booleans instead of ints,
# we will be replacing them here, respectively:

def replace(emp_name_abs):
    for i in range(len(emp_name_abs)):
        if emp_name_abs[i][1] == False:
            emp_name_abs[i][1] = 0
        elif emp_name_abs[i][1] == True:
            emp_name_abs[i][1] = 1

replace(emp_name_abs)

emp_name_abs = [i[1] for i in emp_name_abs]


# ## Let's make our email vocabulary dataset

# I also use pca here in order to limit the quantity of new features created
# from our email data for a better performance in the application of ML later on


def make_dataset(list_of_vocabs, emp_name_abs, df_without_outliers,
				 employees_w_email_dir):
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
    								list_of_vocabs, emp_name_abs, test_size = 0.1)

    vectorizer = TfidfVectorizer(sublinear_tf=True,
                                 stop_words='english',
                                max_df = 0.5)
    
    features_train_transformed = vectorizer.fit_transform(X_train)
    features_test_transformed = vectorizer.transform(X_test)
    
    features_names = np.array(vectorizer.get_feature_names())
    
    selector = SelectPercentile(f_classif, percentile = 0.01)
    selector.fit(features_train_transformed, y_train)

    important_features = selector.get_support(indices=False)
    
    scores = selector.scores_

    scores = scores[important_features]
    
    features_train_transformed =\
    	selector.transform(features_train_transformed)

    features_test_transformed  =\
    	selector.transform(features_test_transformed)

    features_train_transformed = features_train_transformed.toarray()
    features_test_transformed = features_test_transformed.toarray()

    features = np.concatenate((features_train_transformed,
    						   features_test_transformed))
    labels = np.concatenate((y_train, y_test))
    
    scaler = preprocessing.MinMaxScaler()
    rescaled_weight = scaler.fit_transform(features)
    
    features_of_interest = features_names[important_features]

    f_length = len(features_of_interest)

    scores_report =\
	{features_of_interest[i]:scores[i] for i in xrange(f_length)}
    
    return features, labels, features_of_interest, scores_report

features, labels, features_of_interest, scores_report =\
	make_dataset(list_of_vocabs,
				 emp_name_abs,
  				 df_without_outliers,
  				 employees_w_email_dir)    

print
print "Email features of interest scores report: "
print
pp.pprint(scores_report)
print


# ## Adding selected new features to df  

def add_new_features_to_df(df_without_outliers,features_of_interest):
    for new_feature in features_of_interest:
        df_without_outliers[new_feature] = 0
        
add_new_features_to_df(df_without_outliers,features_of_interest)


# ## Adding new features to respective employees 

def add_new_features_to_employees(df_without_outliers,
								  employees_w_email_dir,
								  features_of_interest):
    
    print
    print "Making new Dataset"
    print

    for employee in df_without_outliers.index:
        
        old_features_len = len(df_without_outliers.columns.values.tolist())
        old_features_len = old_features_len - len(features[0])
        
        blanks = np.zeros(len(features_of_interest))
        
        if employee not in employees_w_email_dir:
            
            name_row =\
            df_without_outliers[df_without_outliers.index == employee]
            x = np.append(name_row.iloc[:,:old_features_len].values, blanks)
            df_without_outliers[df_without_outliers.index == employee] = x
            
            
        elif employee in employees_w_email_dir:
            
            n = features[employees_w_email_dir.index(employee)-2]
            name_row =\
            df_without_outliers[df_without_outliers.index == employee]
            x = np.append(name_row.iloc[:,:old_features_len].values, n)
            df_without_outliers[df_without_outliers.index == employee] = x

        
add_new_features_to_employees(df_without_outliers,
							  employees_w_email_dir,
							  features_of_interest)

print 
print "Total number of data points"
print
print df_without_outliers.shape[0]
print
print "Total number of features"
print 
print df_without_outliers.shape[1]
print 
print "Total number of POIs"
print
print df_without_outliers[df_without_outliers.poi == True].shape[0]
print


# ## Scaling


def scaling(df_without_outliers):
    
    scaler = preprocessing.MinMaxScaler()
    df1 = df_without_outliers.copy()
    df1.drop("last_name", axis = 1, inplace = True)
    features = df1.columns.tolist()
    rescaled_weights = scaler.fit_transform(df1)
    df1 = pd.DataFrame(data = rescaled_weights, columns = features)
    
    return df1

df_without_outliers = scaling(df_without_outliers)

print 
print "completed"
print

### Task 4: Try a varity of classifiers 


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html

print 
print "Task 4 & 5"
print


def do_grid_search(df_without_outliers):

    my_dataset = df_without_outliers.reset_index().copy()

    labels = my_dataset.poi.values.tolist()
    features = my_dataset.reset_index().drop("poi", axis = 1).values

    X_train, X_test, y_train, y_test = \
              model_selection.train_test_split(features,labels,test_size = 0.1)

    knb = neighbors.KNeighborsClassifier()
    
    parameters = {'n_neighbors':
                        [5,6,7],
                  'algorithm':
                        ('ball_tree', 'kd_tree', 'brute'),
                  'leaf_size':
                        [10, 20, 30, 40]}

    clf = GridSearchCV(knb, parameters, cv=5)

    clf.fit(X_train, y_train)

    print
    pp.pprint(clf.best_params_)
    print

    return clf.best_params_

def do_ml(df_without_outliers):

    best_params = do_grid_search(df_without_outliers)
    
    my_dataset = df_without_outliers.reset_index().copy()

    labels = my_dataset.poi.values.tolist()
    features = my_dataset.reset_index().drop("poi", axis = 1).values
    
    X_train, X_test, y_train, y_test = \
     		  model_selection.train_test_split(features,labels,test_size = 0.1)
    
    # clf = ensemble.RandomForestClassifier()

    # clf = svm.LinearSVC(kernel = "rbf")
    
    clf =\
     neighbors.KNeighborsClassifier(n_neighbors = best_params['n_neighbors'],
                                         algorithm = best_params['algorithm'],
                                         leaf_size = best_params['leaf_size'])


    clf.fit(X_train, y_train)
    
    confidence = clf.score(X_test, y_test)
    predictions = clf.predict(X_test)
    
    return clf
 
print
print "Training Classifier"  

clf = do_ml(df_without_outliers)

print 
print "completed"
print


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


# ## Setting up data for pickling
# empty_final_features

def find_empty_features(df_without_outliers):

    lista = []

    eff = [(i,set(df_without_outliers[i])) for i in \
           df_without_outliers.columns.tolist()]

    for i in eff:
        lista2 = list(i[1])
        if len(lista2) == 1 and int(lista2[0]) == 0:
            lista.append(i[0])

    return lista

print
print "Empty features:" 
print
pp.pprint(find_empty_features(df_without_outliers))
print

print 
print "Task 6"
print

dp = dp.split('\\')
dp.remove(dp[-1])
dp = '\\'.join(dp)


os.chdir(dp+"\\final_project")

# Pickling our classifier

# with open("my_classifier.pkl", "w") as f:
#     pickle.dump(clf, f)

# # Pickling our dataset

# with open("my_dataset.pkl", "w") as f:
#     df = df_without_outliers.to_dict(orient = "index")
#     pickle.dump(df, f)

# # Pickling our feature_list

# with open("my_feature_list.pkl", "w") as f:
#     df = df_without_outliers
#     lista = df.columns.values.tolist()
#     lista.insert(0, lista.pop(lista.index("poi")))
#     pickle.dump(list(lista), f)

print 
print "completed"
print