#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：deeplearning_python 
@File    ：test.py
@Author  ：Jstan
@Date    ：2021/10/13 23:59 
'''
import numpy as np

a = np.array([2,5,8,9])
b = np.array([[1,2],[3,4]])
l1 = enumerate(a)  # 矩阵变成枚举型
l2= enumerate(b)

# for e in l1:
#     print(e)
'''
(0, 2)
(1, 5)
(2, 8)
(3, 9)
'''
# for e in l2:
#     print(e)
'''
(0, array([1, 2]))
(1, array([3, 4]))
'''

for index,e in l2:
    print(index,e)
'''
0 [1 2]
1 [3 4]

'''
print("****************test**************")
a = np.array([1,2,3])
b = np.array([[1,2],[3,4]])

print(a)
print(b)

print(np.sum(a**2))
print(np.sum(b**2))# 默认所有元素平方求和
print(np.sum(b**2,axis=1))# 每一行的和

print("***********test meshgrid*************")
x0 = np.arange(-2,2.5,0.25)
x1 = np.arange(-2,2.5,0.25)
X,Y = np.meshgrid(x0,x1)

print(X)
print("*****************")
print(Y)
print("******************")
X = X.flatten()
print("X flatten:",X.ndim)#1
Y = Y.flatten()
print(np.array([X,Y]))# ndim=2

