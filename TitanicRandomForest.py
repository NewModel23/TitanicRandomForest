# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 07:05:04 2019

@author: Raúl Guerrero

Applying RandomForestRegressor
i get the inspiration based on Mike Bernico works!


"""

import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import pandas as pd


# os.getcwd()
# Set directory
os.chdir("c:/Users/raulGuerrero/documents/titanic")
# os.listdir()

# Import data
# datos = our train set
datos = pd.read_csv("train.csv")
# datos = pd.read_csv('../input/train.csv')
y = datos.pop("Survived")

# Get the basic statistics of the data
desc = datos.describe()

# Impute age with mean
datos["Age"].fillna(datos.Age.mean(), inplace=True)

# Select only numerical values for fit our model
numeric_variables = list(datos.dtypes[datos.dtypes != "object"].index)

# Build the first model
model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)

# for the model i use only numerical values
model.fit(datos[numeric_variables], y)

model.oob_score_

# So lets get the prediction for a single observation 
# We have an array with the prediction of probability of survival
y_oob = model.oob_prediction_

# get the AUROC (Area Under the Receiver Operating Characteristics)
#   in detinition AUC - ROC curve is a performance measurement for classification problem at various thresholds settings.
roc_auc_score(y, y_oob)

# Now lets make some improvements
# Lets see the describe but for each column
describe_categorical_variables = datos[datos.columns[datos.dtypes == "object"]].describe()

# Data Cleaning
# I will do a data cleaning identifying the columns i don´t need to use for this purpose
# for this sample i decide to drop some columns like Name, Ticket and PassengerID

datos.drop(['Name','Ticket','PassengerId'], axis=1, inplace=True)

#Change the 'Cabin' variable to be only the first letter or none
def clean_cabin(datos):
    try:
        return datos[0]
    except TypeError:
        return "None"

datos["Cabin"] = datos.Cabin.apply(clean_cabin)

categorical_variables = ['Sex','Cabin','Embarked']

for variable in categorical_variables:
    # Fill the missing data with the word "Missing"
    datos[variable].fillna("Missing", inplace=True)
    # Get the dummy variables for that caytegorical
    dummies = pd.get_dummies(datos[variable], prefix=variable)
    # Update datos to include dummies and drop the main variable
    # Concatenate the original dataframe with these dummy variables
    datos = pd.concat([datos, dummies], axis=1)
    # And drop the orginal variable
    datos.drop([variable], axis=1, inplace=True)


# So lets make a model again with the new dataset
# With 100 trees with oob scores calculated throughout it using all the processors and having a random state set to 42

model = RandomForestRegressor(100, oob_score=True, n_jobs=-1, random_state=42)
model.fit(datos, y)

roc_auc_score(y, model.oob_prediction_)

# So we have a 86, its a pretty good model
# RandomForest regressor has many options and there are three many important when trying the make model better
# So i will use RandomForest to do edatosploratory
# I will calculate which variables are more important numerically in the model, the result correspoonds to the order
# of the column names
model.feature_importances_

# To make a clear order i will do a Series that make the match between the column and the importance
feature_importances = pd.Series(model.feature_importances_, index=datos.columns)

# To make this more clear i will order and plotting the Importance
feature_importances.sort_values().plot(kind="barh", figsize=(7,6))

# n_estimator
results = []
n_estimator_options = [30, 50, 100, 200, 500, 1000, 2000]

for trees in n_estimator_options:
    model = RandomForestRegressor(trees, oob_score=True, n_jobs=-1, random_state=42)
    model.fit(datos,y)
    print(trees, "trees")
    roc = roc_auc_score(y,  model.oob_prediction_)
    print("C-stat: roc is: ",roc)
    results.append(roc)
    print("")
    
pd.Series(results, n_estimator_options).plot()

# what the chart tells me is that after thousand trees its pretty flat the roc curve, so we have a good measure of separability


# lets see how the ROC is with max_features
results = []
max_features_options = ["auto",None,"sqrt","log2",0.9,0.2]

for max_features in max_features_options:
    model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=42, max_features=max_features)
    model.fit(datos, y)
    print(max_features, "option")
    roc = roc_auc_score(y, model.oob_prediction_)
    print("c-stat: ", roc)
    results.append(roc)
    print("")
    
pd.Series(results, max_features_options).plot(kind="barh", xlim=(0.85,.88));


# i will do a final thing to test us the minimum samples per leaf
# i will train the model with  of the minimum samples per leaf so we can see the observations in the final node

results = []
min_sample_leaf_options = [1,2,3,4,5,6,7,8,10]

for min_samples in min_sample_leaf_options:
    model = RandomForestRegressor(n_estimators=1000,
                                  oob_score=True,
                                  n_jobs=-1,
                                  random_state=42,
                                  max_features="auto",
                                  min_samples_leaf=min_samples)
    model.fit(datos, y)
    print(min_samples, "option")
    roc = roc_auc_score(y, model.oob_prediction_)
    print("c-stat: ", roc)
    results.append(roc)
    print("")
    
pd.Series(results, min_sample_leaf_options).plot();


# Final model
# From all these optimizations i get the final model
# where there is a 1000 estimators in the model
# the out of bag score on
# njob is -1 (All process)
# random_state = 42
# and we using auto to determine the madatos features which is just every single feature every single variable is considered a huge split
# and it stoping splitting a node when each 5 samples in that node

model = RandomForestRegressor(n_estimators=1000,
                              oob_score=True,
                              n_jobs=-1,
                              random_state=42,
                              max_features="auto",
                              min_samples_leaf=5)


model.fit(datos, y)

roc = roc_auc_score(y, model.oob_prediction_)

print("C-stat: ",roc)