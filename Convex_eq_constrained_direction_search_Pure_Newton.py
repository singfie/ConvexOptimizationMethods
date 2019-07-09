#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:04:56 2019

@author: fietekrutein

A script to find a new direction of an equality constrained 
convex optimization problem using pure Newton's method
"""

import numpy as np

# functions as of example problem

def exp1(x):
    return(np.exp(x[0] + 3*x[1] + x[2] - 0.1))
    
def exp2(x):
    return (np.exp(x[0] - 3*x[1] + 2*x[2] - 0.1))

def exp3(x):
    return (np.exp(-x[0] - x[1] - 2*x[2] - 0.1))
    
def obj(x):
    return (exp1(x) + exp2(x) + exp3(x))

def grad(x):
    i1 = exp1(x) + exp2(x) - exp3(x)
    i2 = 3*exp1(x) - 3*exp2(x) - exp3(x)
    i3 = exp1(x) + 2*exp2(x) - 2*exp3(x)
    return (np.array([i1,i2,i3]))

def hess(x):
    i11 = exp1(x) + exp2(x) + exp3(x)
    i12 = 3*exp1(x) -3*exp2(x) + exp3(x)
    i13 = exp1(x) + 2*exp2(x) + 2*exp3(x)
    
    i21 = 3*exp1(x) - 3*exp2(x) + exp3(x)
    i22 = 9*exp1(x) + 9*exp2(x) + exp3(x)
    i23 = 3*exp1(x) - 6*exp2(x) + 2*exp3(x)
    
    i31 = exp1(x) + 2*exp2(x) + 2*exp3(x)
    i32 = 3*exp1(x) - 6*exp2(x) + 2*exp3(x)
    i33 = exp1(x) + 4*exp2(x) + 4*exp3(x)
    
    return (np.array([[i11,i12,i13],
                      [i21,i22,i23],
                      [i31,i32,i33]]))

# new direction search formulation

def NewDir(x, A):
    g = grad(x)
    h = hess(x)
    h_inv = np.linalg.inv(h)
    d = -np.dot(h_inv, g) + np.dot(np.dot(np.dot(h_inv, A.T), 
                  np.linalg.inv(np.dot(A, np.dot(h_inv, A.T)))), np.dot(A, np.dot(h_inv, g)))
    return (d)

# initialize example
A = np.array([[1,2,0],
              [4,0,5]])
x = np.array([1,1,2/5])

# execute algorithm
NewDir(x,A)

# check null space / whether decent direction
xnew = x + NewDir(x,A)
d2 = NewDir(x,A)
nspace_t = np.dot(A,d2)
np.dot(grad(x).T, d2)
