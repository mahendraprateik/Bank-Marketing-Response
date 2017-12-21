# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 21:32:15 2017

@author: Prateikm
"""

# Importing packages
from Functions import *
import numpy as np
import pandas as pd
import random
import keras
from keras.models import Sequential
from keras.layers import Dense
#from mis680v1 import *
from scipy import stats


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
y = y.reshape(-1,1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and first hidden layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', input_dim = 52))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))


# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# COmpiling the ANN - stochastic gradient descent method
# optimizer = adam is used for stochastic gradiant method for finding weights
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 15, nb_epoch = 100)

# Predict the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred,target_names = ['Class 0', 'Class 1'])
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

rep_df = pd.dataframe()
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


# Accuracy = 0.9088921282798834
# Recall = 0.60149439601494392
# Precision = 0.5255712731229597


from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)


# Calculating vif score
"""from statsmodels.stats.outliers_influence import variance_inflation_factor 
vif = pd.concat([pd.DataFrame(all_input_var), pd.DataFrame([variance_inflation_factor(df[all_input_var].values, ix) for ix in range(df[all_input_var].shape[1])])], axis = 1)
vif.columns = ['variable', 'vif_score']
vif
drop_list  = vif.loc[vif.vif_score>3, 'variable'].tolist()
df = df.drop(drop_list, axis = 1)
"""