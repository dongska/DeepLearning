# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 实现阶跃函数
def step_function(x):
    y = x > 0
    return y.astype(np.int32)

# 实现sigmoid函数
def sigmoid_function(x):
    y = 1/(1 + np.exp(-x))
    return y

# 实现ReLU函数
def relu_function(x):
    return np.maximum(0,x)

# 实现softMax函数 + 防止溢出
def softmax_function(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

# 实现均方误差 损失函数
def mean_squared_error(y,t):
    return np.sum((y-t)**2) / (2 * y.shape[0])  # 

# 实现交叉熵误差 损失函数
def cross_entropy_error(y,t):
    delta = 1e-7    # log运算防止x=0
    if y.ndim == 1: # mini-batch 当为一维时改为 1*n
        t = t.reshape(1,t.size)
        y = y.reshape(1,t.size)
    batch_size = y.shape[0]
    return - np.sum(t * np.log(y + delta)) / batch_size  # 除batch大小

# 实现微分,f在x点处的中心导数
def numerical_diff(f,x):
    h = 1e-4    # delta x 不易过大或过小
    return (f(x + h) - f(x - h)) / (h * 2)

# 实现梯度
def numerical_gradient(f,x):
    grad = np.zeros_like(x)
    h = 1e-4

    for i in range(x.size):
        tmp = x[i]
        # 
        x[i] = tmp + h
        f1 = f(x)
        #
        x[i] = tmp - h
        f2 = f(x)

        grad[i] = (f1 - f2) / (2 * h)
        x[i] = tmp

    return grad

# 梯度法
def gradient_descent(f,init_x,lr,step_num):
    # f为进行最优化的函数，init_x为初始坐标
    # lr为学习率learning rate，step_num为梯度法重复次数
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad 
    return x

def function1(x):
    return 0.01*x**2 + 0.1*x
def function2(x):
    return x[0]**2 + x[1]**2

