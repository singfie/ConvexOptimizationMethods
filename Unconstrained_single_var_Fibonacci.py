#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:59:30 2019

@author: fietekrutein

A script to solve an unconstrained single variable problem using 
the Fibonacci method
"""

# function as per example problem

def gsNext(a0, b0, p):
    return(a0 + p*(b0 - a0))
    
def FMStep(n):
    fib = lambda n: n if n <= 2 else fib(n-1) + fib(n-2)
    return(1 - fib(n)/fib(n+1))
    
# method

def FibonacciMethod(a,b,N,fx):
    aCurrent = a
    bCurrent = b
    
    pN = FMStep(N)
    aNext = gsNext(aCurrent, bCurrent, pN)
    bNext = gsNext(aCurrent, bCurrent, 1-pN)
    fxa = fx(aNext)
    fxb = fx(bNext)
    
    for i in range(N-1,0,-1):
        if fxa < fxb:
            bCurrent = bNext
            bNext = aNext
            fxb = fxa
            aNext = gsNext(aCurrent, bCurrent, pN)
            fxa = fx(aNext)
        else:
            aCurrent = aNext
            aNext = bNext
            fxa = fxb
            bNext = gsNext(aCurrent, bCurrent, 1-pN)
            fxb = fx(bNext)
        print("Iteration {}: {}".format(i+1,(aCurrent+bCurrent)/2))
        pN = FMStep(i)

    pN -= 0.001
    if fxa < fxb:
        bCurrent = bNext
        bNext = aNext
        fxb = fxa
        aNext = gsNext(aCurrent, bCurrent, pN)
        fxa = fx(aNext)
    else:
        aCurrent = aNext
        aNext = bNext
        fxa = fxb
        bNext = gsNext(aCurrent, bCurrent, 1-pN)
        fxb = fx(bNext)

    print("Iteration {}: {}".format(i,(aCurrent+bCurrent)/2))
    return((aCurrent+bCurrent)/2)

# execute

FibonacciMethod(0,2,4,lambda x: x**4 - 14*x**3 + 60*x**2 - 70*x) 