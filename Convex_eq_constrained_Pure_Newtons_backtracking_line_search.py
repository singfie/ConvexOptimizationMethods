#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 21:40:10 2019

@author: fietekrutein

A script for solving an equality constrained convex problem 
using Newton's method with backtracking line search
"""
import numpy as np

# functions as per example problem

def obj(x):
    return (-np.log(x[0]) - np.log(x[1]))
    
def grad(x):
    return (-1/x)

def hess(x):
    return (np.diag(1/np.power(x,2)))

# Newtons method

def NewtonsMethod(x_init,A,grad,hess,alpha, beta, tol, nStop):
    xCurrent = x_init
    nIter=0
    gC = grad(xCurrent)
    hC = hess(xCurrent)
    h_invC = np.linalg.inv(hC)
    
    while True:
        nIter += 1
        t = 1
    
        d = -np.dot(h_invC, gC) + np.dot(np.dot(np.dot(h_invC, A.T), 
                  (1/(np.dot(A, np.dot(h_invC, A.T))))), np.dot(A, np.dot(h_invC, gC)))
        xNext = xCurrent + t*d
        while obj(xNext) > (obj(xCurrent) + alpha*t*np.dot(gC,d)):
            t = beta * t
            xNext = xCurrent + t*d
        
        print("Iteration {}: {}".format(nIter, xNext))
        gC = grad(xNext)
        hC = hess(xNext)
        h_invC = np.linalg.inv(hC)
        if np.linalg.norm(d,2) <= tol or nIter >= nStop:
            break
        
        xCurrent = xNext
        
    return(xCurrent)
    
# initialize problem parameters
    
A = np.array([2,5])
b = 1
alpha = 0.1
beta = 0.7
x_init = np.array([(b/(A[0]+A[1])), (b/(A[0]+A[1]))])

# execute

NewtonsMethod(x_init, A, grad, hess, alpha, beta, tol=10e-5, nStop=20)