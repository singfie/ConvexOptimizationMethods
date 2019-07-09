#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:44:51 2019

@author: fietekrutein

A script to solve a sinusoidal non-linear least squares problem using 
Newton's direction via the Jacobian matrix
"""

import numpy as np
import pandas as pd

def NLLS_Newton(x,y,t): # it was decided to implement for just one iteration, no loop
    m = y.shape[0]
    
    def calc_r(x, y, t):
        return y-x[0]*np.sin(x[1]*t+x[2])
        
    # Jacobian calculation
    def Jacobian(x,y,t):
        J = np.zeros((m,3))
        for i in np.arange(m):
            J[i,0] = -np.sin(x[1]*t[i]+x[2])
            J[i,1] = -t[i]*x[0]*np.cos(x[1]*t[i]+x[2])
            J[i,2] = -x[0]*np.cos(x[1]*t[i]+x[2])
        return J
            
    #Jacobian(x,y,t)   
            
    # Matrix S calculation
    def calcS(x, y,t):
        S = np.zeros((3,3))
        S[0,1] = -np.sum((calc_r(x, y, t)) * t * np.cos(x[1]*t+x[2]))
        S[0,2] = -np.sum((calc_r(x, y, t)) * np.cos(x[1]*t+x[2]))
        S[1,0] = -np.sum((calc_r(x, y, t)) * t * np.cos(x[1]*t+x[2]))
        S[2,0] = -np.sum((calc_r(x, y, t)) * np.cos(x[1]*t+x[2]))
        S[1,1] = np.sum((calc_r(x, y, t)) * x[0] * t**2 * np.cos(x[1]*t+x[2]))
        S[1,2] = np.sum((calc_r(x, y, t)) * t[0] * x[0] * np.cos(x[1]*t+x[2]))
        S[2,1] = np.sum((calc_r(x, y, t)) * t[0] * x[0] * np.cos(x[1]*t+x[2])) 
        S[2,2] = np.sum((calc_r(x, y, t)) * x[0] * np.cos(x[1]*t+x[2])) 
        return S
    
    #calcS(x, y,t)
            
    # Hessian calculation
    def NonLinLSHess(x, y, t):
        J = Jacobian(x,y,t)
        S = calcS(x,y,t)
        H = 2*np.matmul(J.T, J) + S
        return H
    
    #NonLinLSHess(x, y, t)
    
    # calculate next trial solution
    J = Jacobian(x,y,t)
    S = calcS(x,y,t)
    r = calc_r(x,y,t)
    x = x - np.matmul(np.linalg.inv(np.matmul(J.T, J) + S), np.matmul(J.T, r))
    return x

# TESTING #
data = pd.read_excel("data.xlsx", header=1)
y = np.array(data.iloc[:,1])
t = np.array(data.iloc[:,2])           

a = 1
omega = 0.8
theta = 0.1
x = np.array([a,omega,theta]) # initial trial solution
    
# execute

NLLS_Newton(x,y,t)