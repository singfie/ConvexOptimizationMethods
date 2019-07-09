#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:57:54 2019

@author: fietekrutein

A script for solving an unconstrained single variable function using 
the Golden Section method
"""
import numpy as np

# function as per example problem

def gsNext(a0, b0, p):
    return(a0 + p*(b0 - a0))
    
# method
    
def GoldenSection(a,b,fx,tol=0.001):
    aCurrent = a
    bCurrent = b
    
    N = int(np.ceil((np.log(2*tol) - np.log(np.abs(b-a)))/np.log(0.618)))
    
    aNext = gsNext(aCurrent, bCurrent, 0.382)
    bNext = gsNext(aCurrent, bCurrent, 0.618)
    fxa = fx(aNext)
    fxb = fx(bNext)
    
    for i in range(N):
        if fxa < fxb:
            bCurrent = bNext
            bNext = aNext
            fxb = fxa
            aNext = gsNext(aCurrent, bCurrent, 0.382)
            fxa = fx(aNext)
        else:
            aCurrent = aNext
            aNext = bNext
            fxa = fxb
            bNext = gsNext(aCurrent, bCurrent, 0.618)
            fxb = fx(bNext)
            
    return((aCurrent+bCurrent)/2)

# execute

GoldenSection(0,2,lambda x: 2*x**6 + 3*x**4 -12*x, 0.01) 