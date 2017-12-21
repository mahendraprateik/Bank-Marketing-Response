
import os
import subprocess
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn import tree
import pydot
import pandas as pd
import numpy as np

df = pd.read_csv("c:/mis680/bank-additional-full.csv", header=(0), sep=';')


###Convert the categorical to numeric
def tar_encode(df, tar_col):
    df1 = df.copy()
    targets = df1[tar_col].unique()
    job_num = df1["job"].unique()
    marital_num = df1["marital"].unique()
    education_num = df1["education"].unique()
    default_num = df1["default"].unique()
    housing_num = df1["housing"].unique()
    loan_num = df1["loan"].unique()
    contact_num = df1["contact"].unique()
    month_num = df1["month"].unique()
    day_of_week_num = df1["day_of_week"].unique()
    poutcome_num = df1["poutcome"].unique()
    map_to_int = {name: n for n, name in enumerate(job_num)}
    df1["job_num"] = df1["job"].replace(map_to_int)
    map_to_int = {name: n for n, name in enumerate(marital_num)}
    df1["marital_num"] = df1["marital"].replace(map_to_int)
    map_to_int = {name: n for n, name in enumerate(education_num)}
    df1["education_num"] = df1["education"].replace(map_to_int)
    map_to_int = {name: n for n, name in enumerate(default_num)}
    df1["default_num"] = df1["default"].replace(map_to_int)
    map_to_int = {name: n for n, name in enumerate(housing_num)}
    df1["housing_num"] = df1["housing"].replace(map_to_int)
    map_to_int = {name: n for n, name in enumerate(loan_num)}
    df1["loan_num"] = df1["loan"].replace(map_to_int)
    map_to_int = {name: n for n, name in enumerate(contact_num)}
    df1["contact_num"] = df1["contact"].replace(map_to_int)
    map_to_int = {name: n for n, name in enumerate(month_num)}
    df1["month_num"] = df1["month"].replace(map_to_int)
    map_to_int = {name: n for n, name in enumerate(day_of_week_num)}
    df1["day_of_week_num"] = df1["day_of_week"].replace(map_to_int)
    map_to_int = {name: n for n, name in enumerate(poutcome_num)}
    df1["poutcome_num"] = df1["poutcome"].replace(map_to_int)
    map_to_int = {name: n for n, name in enumerate(targets)}
    df1["target"] = df1[tar_col].replace(map_to_int)

    return (df1, targets,job_num)
df2,targets,job_num = tar_encode(df, "y")

###Drop the categorical columns
df2.drop(['y'], axis = 1, inplace = True)
df2.drop(['job'], axis = 1, inplace = True)
df2.drop(['marital'], axis = 1, inplace = True)
df2.drop(['education'], axis = 1, inplace = True)
df2.drop(['default'], axis = 1, inplace = True)
df2.drop(['housing'], axis = 1, inplace = True)
df2.drop(['loan'], axis = 1, inplace = True)
df2.drop(['contact'], axis = 1, inplace = True)
df2.drop(['month'], axis = 1, inplace = True)
df2.drop(['day_of_week'], axis = 1, inplace = True)
df2.drop(['poutcome'], axis = 1, inplace = True)

##correlation
print df2.corr()

features = list(df2.columns[0:20])
y = df2["target"]
X = df2[features]
dt = DecisionTreeClassifier(max_depth=4)
dt.fit(X, y)

tree.export_graphviz(dt, out_file='C:/MIS680/tree.dot', feature_names=X.columns)
(graph,) = pydot.graph_from_dot_file('C:/MIS680/tree.dot')
graph.write_png('C:/MIS680/tree.png')

