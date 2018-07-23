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
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score


def decisionTreeClassifier(X,y):
    clf= tree.DecisionTreeClassifier(criterion="entropy")
    clf=clf.fit(X,y)
    return clf

def overSampling(dataSet):
    ros = RandomOverSampler()
    X= np.array(dataSet.iloc[:,0:9])
    y= np.array(dataSet.iloc[:,9])
    X_ros, y_ros = ros.fit_sample(X, y)
    dataSet=np.concatenate((X_ros,y_ros[:,None]),axis=1)
    dataSet =pd.DataFrame(dataSet)
    return dataSet

filename = r'ProcessData-Encode.xlsx'
dataSet = pd.read_excel(filename)
dataSet= overSampling(dataSet)
dataTrain,dataTest = train_test_split(dataSet, shuffle=True, test_size=0.30)
X= np.array(dataTrain.iloc[:,0:9])
y= np.array(dataTrain.iloc[:,9])
model = decisionTreeClassifier(X,y)
predictionsTest = model.predict(dataTest.iloc[:,0:9])
predictionsTrain =model.predict(dataTrain.iloc[:,0:9])
accuracyTest = accuracy_score(dataTest.iloc[:,9], predictionsTest)
accuracyTrain= accuracy_score(dataTrain.iloc[:,9], predictionsTrain)
f1ScoreTest =f1_score(dataTest.iloc[:,9], predictionsTest, average='weighted')
f1ScoreTrain=f1_score(dataTrain.iloc[:,9], predictionsTrain, average='weighted')


#figure = plt.figure().add_subplot(111)
#figure.scatter(dataTest.iloc[:,9],predictions)
#plt.show()
#plt.scatter(dataTest.iloc[:,9], predictions)
#plt.xlabel("True Values")
#plt.ylabel("Predictions")
#plt.show()
print "accuracy in predicting Data Train : ",accuracyTrain
print "accuracy in predicting Data Test  : ",accuracyTest
print "f1 score in predicting Data Train : ",f1ScoreTrain
print "f1 score in predicting Data Test : ",f1ScoreTest

