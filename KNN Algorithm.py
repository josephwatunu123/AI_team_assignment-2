# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 18:39:01 2022

@author: 134775 HAKEEM ALAVI

KNN - Predict whether a person has diabetes or not
"""
# The dependencies
import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# The diabetes dataset
dataset = pd.read_csv('diabetes.csv')
print(dataset.head())

# Will not directly take data if the mentioned columns have 0
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for column in zero_not_accepted:
    dataset[column]= dataset[column].replace(0,np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN,mean)
# The dependant (y) and independent (X) variabes
X = dataset.iloc[:,0:8]
y = dataset.iloc[:,8]

# Training the respective variables
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.2,)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')

classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test,y_pred)
print("")
print("Confusion matrix:")
print(cm)

# F1 score
print("")
print("F1 score is",f1_score(y_test,y_pred))

# Accuracy
print("")
print("Accuracy is",accuracy_score(y_test,y_pred))

# Results Interpretation (Confusion Matrix):
# Out of 94 people that did not have diabetes (True Positive), the prediction said 13 have diabetes (False Positive).
# Out of 32 people that had diabetes (False Negative), the prediction said 15 did not have diabetes (True Negative).

