#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:11:23 2019

@author: fietekrutein

A script for a second order cone problem using cvxpy package
"""

import numpy as np
import cvxpy as cp

# initialize values

Q = np.array([[2,1],[1,1]])
ahat = np.array([1,1])
P = np.sqrt(Q)
Pinv = np.linalg.inv(np.sqrt(Q))
c = np.array([-2,-3])
b = 1

# Define and solve in cvxpy
n = 2
x = cp.Variable(n)
objective = cp.Minimize(c*x)
constraints = [cp.pnorm(P*x) <= (b-ahat*x),
               x >= 0]
prob = cp.Problem(objective, constraints)
prob.solve()

result = prob.solve()
# The optimal value for x is stored in `x.value`.
print("The optimum is reached at: ", result)
print("The optimal values for x are: ", x.value)