#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：deeplearning_python 
@File    ：sigmoid.py
@Author  ：Jstan
@Date    ：2021/10/13 21:30 
'''
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

X = np.arange(-5.0,5.0,0.1)
Y = sigmoid(X)

plt.plot(X,Y)

plt.xlabel("x")
plt.ylabel("y")
plt.ylim(-0.1,1.1)
plt.title("sigmoid")

plt.show()