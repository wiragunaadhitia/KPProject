# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 06:35:15 2018

@author: HP
"""

from pandas import DataFrame
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score


def decisionTreeClassifier(X,y):
    clf= tree.DecisionTreeClassifier()
    clf=clf.fit(X,y)
    return clf

filename = r'ProcessData-Encode.xlsx'
dataSet = pd.read_excel(filename)
dataTrain,dataTest = train_test_split(dataSet, shuffle=True, test_size=0.30)
X= np.array(dataTrain.iloc[:,0:9])
y= np.array(dataTrain.iloc[:,9])
model = decisionTreeClassifier(X,y)
prediksi = model.predict(dataTest.iloc[:,0:9])
accuracy = accuracy_score(dataTest.iloc[:,9], prediksi)
print accuracy
