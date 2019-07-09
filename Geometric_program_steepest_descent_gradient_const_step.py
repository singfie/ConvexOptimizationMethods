#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 9 11:36:30 2019

@author: fietekrutein

A script for solving a geometric program using
steepest descent gradient search with constant step size
"""

import numpy as np

# method

def GradSearchGeom(yold, t, n, a, c, tol=10e-3):
    i = 1
    stop = 0
    f = np.empty([1,1])
    while stop == 0:
        numvec1 = np.exp(yold[0]+yold[1]) - np.exp(-yold[0])
        numvec2 = np.exp(yold[0]+yold[1]) - np.exp(-yold[1])
        denom = np.exp(yold[0]+yold[1]) + np.exp(-yold[0]) + np.exp(-yold[1])
        grad = np.array([numvec1/denom, numvec2/denom])
        f = np.append(f, np.log(denom))
        if np.linalg.norm(grad, 2) <= tol:
            stop = 1
            pstar = np.log(denom)
            ystar = yold
            print('p*:', pstar, 'y*:', ystar, 'iterations:', i)
            print('function values:', f)
        else:
            # step size t steepest descent gradient search
            ynew = yold-t*grad
            yold = ynew
            i = i+1 # increase counter
            
# initialize
n = 2
t = 0.1
yold = np.ones(n) # initial trial solution
a = np.array([[1,1],[0,-1],[-1,0]]) # exponent
c = np.array([1,1,1])            
            
# execute
GradSearchGeom(yold, t,n,a,c,tol=10e-3)