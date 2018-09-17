#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:02:44 2018

@author: saroj
"""
from statistics import mean 
import numpy as np
import matplotlib.pyplot as plt

xs = np.array([1,2,3,4,5,6], dtype = np.float64) #data type conversion from integer to float 
ys = np.array([1,5,4,6,7,9], dtype = np.float64)
plt.scatter(xs,ys) #how the data looks on graph

def best_fit_slope_and_intercept(xs,ys):
    m = ( ((mean(xs)*mean(ys)) - mean(xs*ys))/
         ((mean(xs)*(mean(xs))) - mean(xs*xs)) )
    b = mean(ys) - m*mean(xs)
    return m,b
m,b = best_fit_slope_and_intercept(xs, ys)
print(m,b)
regression_line = [(m*x)+b for x in xs]
predict_x = 7 #lets make a prediction for x= 7
predict_y =(m*predict_x) + b
print(predict_y)
plt.plot(xs,regression_line)
plt.scatter(predict_x, predict_y , color='r')