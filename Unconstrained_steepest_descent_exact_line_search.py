#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:48:56 2019

@author: fietekrutein

A script to solve an unconstrained problem using
steepest descent fradient search with exact line search
"""
import numpy as np
import matplotlib.pyplot as plt

# functions as per example problem

def obj(x,Q):
    return 0.5*np.matmul(x, np.matmul(Q,x))
    
def gx2(x,Q):
    return np.matmul(Q,x)

# method
    
def GradDescent2(x,Q,gx2,rec,tol=10e-5):
    xCurrent = x
    gxVal = gx2(xCurrent,Q)
    rec.append(obj(xCurrent,Q))
    while True:
        xNext = xCurrent - ((xCurrent[0]**2 + gamma[i]**2*xCurrent[1]**2)/
                            (xCurrent[0]**2 + gamma[i]**3*xCurrent[1]**2))*gxVal
        gxNextVal = gx2(xNext, Q)
        rec.append(obj(xNext,Q))
        
        if (np.linalg.norm(gxNextVal,2) <= tol) == True:
            break
        xCurrent = xNext
        gxVal = gxNextVal
        
    return xNext, rec

# initialize problem parameters
gamma = np.array([0.1, 1, 2, 10])
rec = np.empty(4, dtype=np.object)
    
# execute 

for i in np.arange(0,4):
    x = np.array([gamma[i], 1])
    Q = np.array([[1,0],[0,gamma[i]]])
    rec[i] = []
    GradDescent2(x, Q, gx2, rec[i], tol=10e-5)
    
# plot
    
plt.subplot(2,2,1)
plt.plot(rec[0])
plt.xlabel('iterations')
plt.ylabel('objective value')
plt.subplot(2,2,2)
plt.plot(rec[1])
plt.xlabel('iterations')
plt.subplot(2,2,3)
plt.plot(rec[2])
plt.xlabel('iterations')
plt.ylabel('objective value')
plt.subplot(2,2,4)
plt.plot(rec[3])
plt.xlabel('iterations')

plt.show()