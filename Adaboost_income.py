# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 08:37:06 2024

@author: Rohit Chavan
"""

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('C:/Data_Science_2.0/Adaboost/income.csv')
data.columns
data.head

# let us split the data in input and output
X = data.iloc[:, 0:6]
y = data.iloc[:, 6] 

# train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Creata adaboost classifier
ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=1)
# n_estimators = number of weak learners
# learning_rate, it contributes weights of weak learners, by default

# train the model
model = ada_model.fit(X_train, y_train)

# predict the result
y_pred = model.predict(X_test)
print("accuracy : ", metrics.accuracy_score(y_test, y_pred))

# let us try for another base model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

# here base model is changed
ada_model = AdaBoostClassifier(n_estimators=50, estimator=lr, learning_rate=1)
model = ada_model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
