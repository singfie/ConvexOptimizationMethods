#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:39:07 2019

@author: fietekrutein

A script for solving logistic regression using steepest descent gradient search 
with constant step size
"""

import numpy as np

# function as per exercise
    
def grad_log(a):
    c = np.zeros([n,4])
    for i in np.arange(n):
        c[i,] = (y_raw[i]-(1/(1+np.exp(np.dot(-x_raw[i,], a)))))* x_raw[i,]
    return np.sum(c, axis=0)

# method
    
def GradDescent(a,grad_log, tol=10):
    aCurrent = a
    gaVal = grad_log(aCurrent)
    i = 0
    while True:
        aNext = aCurrent - t*gaVal
        gaNextVal = grad_log(aNext)
        
        i=i+1
        if (i >= tol) == True:
            break
        aCurrent = aNext
        gaVal = gaNextVal
        
    return aNext

# initialize
x_raw = np.array([[1, 380, 3.61, 3],
             [1, 660, 3.67, 3],
             [1, 800, 4, 1],
             [1, 640, 3.19, 4],
             [1, 520, 2.93, 4],
             [1, 760, 3, 2]])
y_raw = np.array([0,1,1,1,0,1])

t = 0.1
k = 10 # iterations
        
n = len(y_raw)
a = np.array([0,0,0,0]) # starting point for a parameters

# execute

GradDescent(a, grad_log, tol=10)  