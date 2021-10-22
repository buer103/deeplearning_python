#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：deeplearning_python 
@File    ：gradient_method.py
@Author  ：Jstan
@Date    ：2021/10/17 15:48 
'''
# 梯度下降法的简单实现

import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    '''
    :param f: 优化函数
    :param init_x: 初始x值
    :param lr: 学习率   learning rate
    :param step_num: 迭代次数
    :return: 迭代中的历史变化
    '''
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    print(np.array(x_history))
    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x,lr=lr)

plt.plot( [-5, 5], [0,0], '--b') # 虚线 + 颜色
plt.plot( [0,0], [-5, 5], '--r')
plt.plot(x_history[:,0], x_history[:,1], 'o')# 表示 点

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()