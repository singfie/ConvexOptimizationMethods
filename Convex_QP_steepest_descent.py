#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:02:29 2019

@author: fietekrutein

A script to solve an unconstrained QP problem using 
the steppest gradient descent method
"""

import numpy as np

# function as per exercise

def gx(x,Q,b):
    return(np.dot(Q,x) - b)
    
# method
def GradDescent(x,gx,Q,b,tol=10e-3):
    xCurrent = x
    gxVal = gx(xCurrent,Q,b)
    
    while True:
        xNext = xCurrent - (np.dot(gxVal,gxVal)/np.dot(np.dot(gxVal,Q), gxVal))*gxVal
        gxNextVal = gx(xNext,Q,b)
        
        if np.all(np.abs(gxNextVal) <= tol) == True:
            break
        xCurrent = xNext
        gxVal = gxNextVal
        
    return xNext

# initialize

x = np.array([0,0])
Q = np.array([[1,-1],[-1,2]])
b = np.array([0,-2])

# execute

GradDescent(x,gx,Q,b)