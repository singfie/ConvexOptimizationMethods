#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:51:23 2019

@author: fietekrutein

A script to solve an uncsontrained problem using the steepest descent gradient
method with backtracking line search
"""

import numpy as np

# functions as per example 

def q10Fx(x):
   A1 = np.array([[1,1,-1],[3,-3,0]])
   b1 = np.array([-0.1,-0.1,-0.1])
   c1 = np.array([1,1,-1])

   A2 = np.array([[1,1],[3,-3]])
   b2 = np.array([-0.1,-0.1])
   c2 = np.array([3,-3])

   grads = []
   grads.append(np.sum(c1*np.exp(np.dot(x,A1)+b1)))
   grads.append(np.sum(c2*np.exp(np.dot(x,A2)+b2)))

   return(np.array(grads))
   
def q10F(x):
   A = np.array([[1,1,-1],[3,-3,0]])
   b = np.array([-0.1,-0.1,-0.1])

   return(np.sum(np.exp(np.dot(x,A)+b)))
   
# method
    
def GradDescentBack(x,tol=10e-5):
    xCurrent = x
    gxVal = q10Fx(xCurrent)
    i = 0
    while True:
        t = 1
        xNext = xCurrent - t*gxVal
        while q10F(xNext) > (q10F(xCurrent) + alpha*t*np.dot(gxVal,-gxVal)):
            t = beta * t
            xNext = xCurrent - t*gxVal
        gxNextVal = q10Fx(xNext)
        
        if (np.linalg.norm(gxNextVal,2) <= tol) == True:
            break
        
        xCurrent = xNext
        gxVal = gxNextVal
        i=i+1
        
    return xNext  , i   
        
# initialize 
alpha = 0.1
beta = 0.7
x = np.array([0,0])

# execute

GradDescentBack(x, tol=10e-5)