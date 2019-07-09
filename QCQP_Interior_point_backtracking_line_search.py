#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 23:14:54 2019

@author: fietekrutein

A script for solving a QCQP, using the log-barrier interior point method with backtracking line search
"""

import numpy as np
import cvxpy as cp
import math

# functions as per example problem 

def obj(x,Q,p,r): # original objective
    return ((0.5*np.dot(np.dot(x,Q),x.T) + np.dot(p,x.T) + r)[0][0])
    
def grad(x,Q,p): # original gradient
    return (np.dot(x,Q).T + p)

def hess(Q): # original hessian 
    return (Q)

def logbar(x,Q,p,r): # log barrier objective
    return (eta * obj(x,Q,p,r) - np.log(x[0][0] - x[0][1] + 2)
            - np.log(-x[0][0]-3*x[0][1]+5) - np.log(-x[0][0]**2 - x[0][1]**2 + 2*x[0][1] + 1))
    
def gradg3(x): # third gradient
    return (2*x - np.array([[0,2]]).T)

# Interior point method

def InteriorPoint(x_init, Q, p, r, alpha, beta, eta, nu, l, tol):
    # fixed constraint gradients
    gradg1 = np.array([[-1,1]])
    gradg2 = np.array([[1,3]])
    hessg3 = np.array([[2,0],
                       [0,2]])
    lIter = 0
    xCurrent = x_init
    while True:
        lIter += 1
        stop = 0
        nIter = 0
        
        while (stop == 0):  
            nIter +=1
            g1 = -xCurrent[0][0] + xCurrent[0][1] - 2 # constraint 1
            g2 = xCurrent[0][0] + 3*xCurrent[0][1] - 5 #constraint 2
            g3 = xCurrent[0][0]**2 + xCurrent[0][1]**2 - 2*xCurrent[0][1] - 1 #constraint 3
            # gradient and hessian of P3(eta)
            gradient = eta * grad(xCurrent,Q,p.T) + (1/-g1)*gradg1.T + (1/-g2)*gradg2.T + (1/-g3)*gradg3(xCurrent.T)    
            hessian = eta * hess(Q) + (1/(-g1)**2)*np.matmul(gradg1.T, gradg1) 
            + (1/(-g2)**2)*np.dot(gradg2.T, gradg2) 
            + (1/(-g3)**2)*np.dot(gradg3(xCurrent).T, gradg3(xCurrent))
            + (1/-g3)*hessg3
            
            t = 1
            d = np.dot(np.linalg.inv(hessian),gradient)
            xNext = xCurrent - t*d.T
            
            # check whether log goes to infinity at evaluated t
            while math.isnan(logbar(xNext,Q,p,r)): 
                t = 0.98 * t
                xNext = xCurrent - t*d.T
            # implement backtracking line search 
            while logbar(xNext,Q,p,r) > (logbar(xCurrent,Q,p,r) + alpha*t*(np.dot(gradient.T,d).item(0))):
                t = beta * t
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
    
# initialize problem parameters
alpha = 0.01
beta = 0.7
eta = 1
nu = 2
l = 3
x_init = np.array([[1,1]])
Q = np.array([[1,0],[0,1]])
p = np.array([[-2,-2]])
r = 0

# execute

InteriorPoint(x_init, Q, p, r, alpha, beta, eta, nu, l, tol=10e-5)

# cvxpy script for comparison

# Define and solve
n = 2
x = cp.Variable(n)
objective = cp.Minimize(0.5*cp.quad_form(x, Q) + p*x + r)
constraints = [-x[0]+ x[1] <= 2,
               x[0] + 3*x[1] <= 5,
               x[0]**2 + x[1]**2 -2*x[1] <= 1]
prob = cp.Problem(objective, constraints)
prob.solve()

result = prob.solve()
# The optimal value for x is stored in `x.value`.
print("The optimum is reached at: ", result)
print("The optimal values for x are: ", x.value)