#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：deeplearning_python 
@File    ：gradient_2d.py
@Author  ：Jstan
@Date    ：2021/10/14 0:04 
'''
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D    # 目的都是生成具有三维格式的对象 Axes3D

# 函数二维梯度计算
def _numerical_gradient_no_batch(f,x):
    # 0.0001  用于近似求解导数
    h = 1e-4
    # 定义梯度矩阵 存储 中间计算的梯度
    grad = np.zeros_like(x)
    pass

# 处理不同维度的梯度矩阵
def numerical_gradient(f,X):
    pass
'''
np.sum(a)
np.sum(a, axis=0) ------->列求和
np.sum(a, axis=1) ------->行求和
'''
# 定义待求梯度函数
def function_2(x):
    pass

# 近似切线
def tangent_line(f,x):
    pass

if __name__=='__main__':
    pass