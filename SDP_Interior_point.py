#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 18:59:11 2019

@author: fietekrutein

A script for a Semi definite programming problem using the
log-barrier interior point method
"""

import numpy as np
import cvxpy as cp

# define functions

def obj(x): # original objective
    return x
    
def grad(x): # original gradient
    return np.array([1])

def logbar(x, F0, F1): # log barrier objective
    return (eta * obj(x) - np.log(np.linalg.det(F(x, F0, F1))))

def F(x, F0, F1):
    return (F0 + x*F1)

# Interior point method

def InteriorPoint(x_init, F0, F1, alpha, beta, eta, nu, l, tol):
    lIter = 0
    xCurrent = x_init
    while True:
        lIter += 1
        stop = 0
        nIter = 0
        
        while (stop == 0):  
            nIter +=1
            neg_F = F(xCurrent, F0, F1)
            in_neg_F = np.linalg.inv(neg_F)
            # gradient and hessian of P3(eta)
            gradient = eta * grad(xCurrent) + np.trace(np.dot(in_neg_F, F1))
            hessian = np.trace(np.dot(np.dot(in_neg_F, F1), np.dot(in_neg_F, F1))) #SDP 
            
            t = 1
            d = (1/hessian)*gradient
            xNext = xCurrent - t*d.T
 
            print("IP Iteration {}, Newton Iteration {}: {}".format(lIter, nIter, xNext))

            if np.linalg.norm((xCurrent - xNext),2) <= tol:
                stop = 1
                xStar = xNext
                break
            else:
                xCurrent = xNext
        
        xCurrent = xStar
        if (l/eta <= tol):
            print("Optimal value found at IP iteration {}: {}".format(lIter, xStar))
            break
        else:
            eta = nu*eta
        
    return(xCurrent)
     
# initialize the example problem
    
alpha = 0.01
beta = 0.7
eta = 1
nu = 2
l = 1
c = 1
x_init = np.array([2])
F0 = np.array([[-1, np.sqrt(2)],
                [np.sqrt(2), -1]])
F1 = np.array([[1, 0],
                [0, 1]])
    
# solve

InteriorPoint(x_init, F0, F1, alpha, beta, eta, nu, l, tol=10e-5)


# Define and solve comparison to cvxpy
n = 1
x = cp.Variable(n)
objective = cp.Minimize(x)
constraints = [F0 + x*F1 >> 0]
prob = cp.Problem(objective, constraints)
prob.solve()

result = prob.solve()
# The optimal value for x is stored in `x.value`.
print("The optimum is reached at: ", result)
print("The optimal values for x are: ", x.value)