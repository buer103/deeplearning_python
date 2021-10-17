#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：deeplearning_python 
@File    ：gradient_3d.py
@Author  ：Jstan
@Date    ：2021/10/17 15:39 
'''
'''
f(x0,x1) = x0**2+x1**2
的图像的实现
'''
import numpy as np
import matplotlib.pylab as plt

fig = plt.figure()
x0 = np.arange(-3,3,0.5)
x1 = np.arange(-3,3,0.5)
x0,x1 = np.meshgrid(x0,x1)
print(x0)
print("***********************************")
print(x1)

#
def function(x0,x1):
    return x0**2+x1**2

fx = function(x0,x1)
ax = plt.axes(projection='3d')
ax.plot_surface(x0, x1, fx, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('fx')
plt.show()
