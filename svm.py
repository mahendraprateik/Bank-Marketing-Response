# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 02:01:42 2017

@author: Prateikm
"""

# Grid Search


# Importing packages
from Functions import *
import numpy as np
import pandas as pd
import random
#import keras
#from keras.models import Sequential
from keras.layers import Dense
#from mis680v1 import *
from scipy import stats
from numpy import *
from collections import defaultdict
# Importing the dataset
dataset = pd.read_csv("bank-additional-full.csv")
dataset.shape

# Function for creating 2000 random missing values
def randomint(df):
    n = 2000
    setOfNumbers = set()
    while len(setOfNumbers) < n:
        setOfNumbers.add(random.randint(0, 40000))
    for i in setOfNumbers:
        df.loc[i,'age'] = np.NaN
    return df

# Creating 2000 random na values
dataset = randomint(dataset)

# Replacing missing values with median - using function from python file - Functions.py
VT = variableTreatment()
VT.replace_missing(dataset, 'median').head()
dataset.info()

# Encoding target variable
VT.encode_target(dataset, 'Response')

# Converting Categorical to dummy
f = dataset['Response']
dataset.drop('Response', axis = 1, inplace = True)
dataset = pd.get_dummies(dataset, drop_first = True)
dataset['Response'] = f
dataset['Response'] = dataset['Response'].astype(int)

# Removing duration as it is highly correlated
dataset.drop('duration', axis = 1, inplace = True)

# Separating target and independent variables
X = dataset.iloc[:, :52].values
y = dataset.iloc[:, 52].values
#y = y.reshape(-1,1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred,target_names = ['Class 0', 'Class 1'])

# Area under curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred,target_names = ['Class 0', 'Class 1'])
rep_df = pd.DataFrame()
def report2dict(cr):
    # Parse rows
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)

    # Store in dictionary
    measures = tmp[0]

    D_class_data = defaultdict(dict)
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    return D_class_data

target_names = ['Class 0', 'Class 1']
rep_df = pd.DataFrame(report2dict(classification_report(y_test, y_pred, target_names=target_names)))


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': 10.**arange(-7,7,3), 'kernel': ['rbf'], 'gamma': 10.**arange(-7,7,3)}]

#parameters = [{'C': 10.**arange(-3,3,2), 'kernel': ['linear']},
 #             {'C': 10.**arange(-3,3,2), 'kernel': ['rbf'], 'gamma': 10.**arange(-3,3,2)}]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'recall',
                           cv = 2,
                           verbose = 20,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_