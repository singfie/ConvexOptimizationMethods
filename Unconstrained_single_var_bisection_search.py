#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:55:29 2019

@author: fietekrutein

A script for solving a single variable unconstrained problem using 
bisection local search method
"""

import numpy as np

# function as per example 
def fx(x):
    return(2*x**6 + 3*x**4 -12*x)

# bisection method
def BiSection(a,b,fx,tol=0.001):
    while np.abs(b-a) > 2*tol:
        xHat = (a+b)/2
        fxVal = fx(xHat)
    
        (a,b) = (xHat,b) if fxVal < 0 else (a,xHat)

    return(xHat)

# execute

BiSection(0,2.4,lambda x: x**3 - 3*x**2 + 4*x - 2, 0.01)    

