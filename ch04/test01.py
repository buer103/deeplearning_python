#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：deeplearning_python 
@File    ：test01.py
@Author  ：Jstan
@Date    ：2021/10/19 22:43 
'''
import numpy as np
'''
randn函数返回一个或一组样本，具有标准正态分布
以0为均值、以1为标准差的正态分布，记为N（0，1）
'''
print(np.random.randn(2,3))

#softmax 函数实现
def softmax(x):
    if x.ndim ==2:
        x = x.T
        x = x-np.max(x,axis=0)
        y = np.exp(x)/np.sum(np.exp(x),axis=0)
        return y.T
    x -= np.max(x) # 溢出对策
    return np.exp(x)/np.sum(np.exp(x))