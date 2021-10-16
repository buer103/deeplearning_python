#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：deeplearning_python 
@File    ：gradient_1d.py
@Author  ：Jstan
@Date    ：2021/10/13 23:27 
'''
import numpy as np
import matplotlib.pylab as plt

# 导数近似值
def numerical_diff(f,x):
    h = 1e-4 #0.0001
    return (f(x+h)-f(x-h))/(2*h)

# 定义的曲线
def function_1(x):
    return 0.01*x**2+0.1*x

def tangent_line(f,x):
    k = numerical_diff(f,x)
    print("近似斜率",k)
    b = f(x)-k*x
    return lambda t:k*t+b

x= np.arange(0.0,20.0,0.1)
y= function_1(x)

tf = tangent_line(function_1,5)
y2 = tf(x)

plt.plot(x,y,label="quxian")
plt.plot(x,y2,label="zhixian")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gradient")
plt.legend(loc="upper left")
plt.show()