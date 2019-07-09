#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:41:54 2019

@author: fietekrutein

A script to solve an unconstrained problem using pure Newton's method 
with backtracking line search
"""
import numpy as np

# functions as per example problem
        
def q3Fx(x):
    c1 = 2*(x[0]+10*x[1])+40*(x[0]-x[3])**3
    c2 = 20*(x[0]+10*x[1]) + 4*(x[1]-2*x[2])**3
    c3 = 10*(x[2]-x[3]) - 8*(x[1]-2*x[2])**3
    c4 = -10*(x[2]-x[3])-40*(x[0]-x[3])**3
    return np.array([c1,c2,c3,c4])
   
def q3F(x):
    return (x[0]+10*x[1])**2 + 5*(x[2]-x[3])**2 + (x[1]-2*x[2])**4 + 10* (x[0]-x[3])**4

# method
    
def GradDescentBack(x,tol=10e-5):
    xCurrent = x
    gxVal = q3Fx(xCurrent)
    i = 0
    while True:
        t = 1
        xNext = xCurrent - t*gxVal
        while q3F(xNext) > (q3F(xCurrent) + alpha*t*np.dot(gxVal,-gxVal)):
            t = beta * t
            xNext = xCurrent - t*gxVal
        gxNextVal = q3Fx(xNext)
        
        if (np.linalg.norm(gxNextVal,2) <= tol) == True:
            break
        
        xCurrent = xNext
        gxVal = gxNextVal
        i=i+1
        
    return xNext  ,i      

# initialzie problem parameters
A = np.array([[1082, 20, 0, -480],
          [20, 212, -24, 0],
          [0, -24, 58, -10],
          [-480, 0, -10, 490]])        
inv_A = np.linalg.inv(A)
x = np.array([3,-1,0,1])
g = np.array([306, -144, -2, -310])
Xn = x - np.matmul(inv_A, g)
        
alpha = 0.2
beta = 0.3
x = np.array([3,-1,0,1])

# execute

GradDescentBack(x, tol=10e-5)