#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：deeplearning_python 
@File    ：test.py
@Author  ：Jstan
@Date    ：2021/10/9 23:26 
'''

import numpy as np

# A= np.array([[1,2,3],[4,5,6]])
# B =np.array([[1,2],[3,4],[5,6]])
#
# C = np.dot(A,B)
#
# print(C)


# softmax 函数的实现

def softmax(x):
    c = np.max(x)

    exp_x = np.exp(x-c)    # 溢出对策
    sum_exp_x = np.sum(exp_x)
    y = exp_x/sum_exp_x

    return y

a= np.array([0.3,2.9,4.0])
y = softmax(a)
print(y)
#[0.01821127 0.24519181 0.73659691]

print(np.sum(y))  # 1.0
# soft 函数的输出在0.0 到1.0 之间  并且sum=1
