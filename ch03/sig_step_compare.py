#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：deeplearning_python 
@File    ：sig_step_compare.py
@Author  ：Jstan
@Date    ：2021/10/13 21:43 
'''
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def step_function(x):
    return np.array(x>0,dtype=int)

x = np.arange(-5.0,5.0,0.1)
y1 = sigmoid(x)
y2 = step_function(x)

plt.plot(x,y1,label="sigmoid",color="red")
plt.plot(x,y2,linestyle="--",label="step",color="blue")
plt.xlabel("x")
plt.ylabel("y")
plt.ylim(-0.1,1.1)
plt.title("step_sigmoid_compare")
plt.legend(loc="upper left")

plt.show()