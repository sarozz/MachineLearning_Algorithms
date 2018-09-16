#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 20:45:08 2018

@author: saroj
"""
import numpy as np
import pandas as pd 
import math
import quandl  #quandl provides datasets 
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression 
df = quandl.get("WIKI/GOOGL") #this is the dataset of stock proces

df = df [['Adj. Open', 'Adj. High','Adj. Low','Adj. Close','Adj. Volume']] #making a dataframe of only useful features
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df ['Adj. Close'] * 100.0 #calculating the highest percentage before closing 

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df ['Adj. Open'] * 100.0 #calcualting the percentage changes in a day

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']] #these are the only features that are needed to predict the next stock price 

forecast_col = 'Adj. Close' 

#we need to fill the NaN values, in machine learning we cannot risk the loss of any data, so we fill it
df.fillna(-99999, inplace = True) #best way to fill NaN places

forecast_out = int(math.ceil(0.01 * len(df))) #math.ceil get anything and takes it to the ceiling, gives the ceiling value. Since math.ciel returns the float values int is defined
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out) #shifting the column negatively, means going up
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1)) #our featres are everything except the label and np.array makes the array 

y = np.array(df['label'])

X = preprocessing.scale(X) #feature scalling

#X = X[:-forecast_out+1]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2) #0.2 is 20%

clf = LinearRegression()

clf.fit(X_train, y_train)  #training the regression model

accuracy = clf.score(X_test, y_test)

print(accuracy)
