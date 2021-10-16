#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：deeplearning_python 
@File    ：step_function.py
@Author  ：Jstan
@Date    ：2021/10/13 21:37 
'''
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x>0,dtype=int)

X = np.arange(-5.0,5.0,0.1)
Y = step_function(X)

plt.plot(X,Y)

plt.xlabel("x")
plt.ylabel("y")
plt.ylim(-0.1,1.1)
plt.title("Step_function")

plt.show()