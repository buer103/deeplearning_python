#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：deeplearning_python 
@File    ：relu.py
@Author  ：Jstan
@Date    ：2021/10/13 21:57 
'''
import numpy as np
import matplotlib.pylab as plt
def relu(x):
    return np.maximum(x,0)



# 展示relu 函数图像
X3 = np.arange(-5.0,5.0,0.1)
Y3 = relu(X3)

plt.plot(X3,Y3)
plt.ylim(-1.0,5.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("ReLU")
plt.show()