# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ʵ�ֽ�Ծ����
def step_function(x):
    y = x > 0
    return y.astype(np.int32)

# ʵ��sigmoid����
def sigmoid_function(x):
    y = 1/(1 + np.exp(-x))
    return y

#
def sigmoid_grad_function(x):
    return (1.0 - sigmoid_function(x)) * sigmoid_function(x)

# ʵ��ReLU����
def relu_function(x):
    return np.maximum(0,x)

# ʵ��softMax���� + ��ֹ���
# def softmax_function(x):
#     c = np.max(x)
#     exp_x = np.exp(x - c)
#     sum_exp_x = np.sum(exp_x)
#     y = exp_x / sum_exp_x
#     return y
def softmax_function(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # ����Բ�
    return np.exp(x) / np.sum(np.exp(x))

# ʵ�־������ ��ʧ����
def mean_squared_error(y,t):
    return np.sum((y-t)**2) / (2 * y.shape[0])  # 

# ʵ�ֽ�������� ��ʧ����
def cross_entropy_error(y,t):
    delta = 1e-7    # log�����ֹx=0
    if y.ndim == 1: # mini-batch ��Ϊһάʱ��Ϊ 1*n
        t = t.reshape(1,t.size)
        y = y.reshape(1,t.size)
    batch_size = y.shape[0]
    return - np.sum(t * np.log(y + delta)) / batch_size  # ��batch��С

# ʵ��΢��,f��x�㴦�����ĵ���
def numerical_diff(f,x):
    h = 1e-4    # delta x ���׹�����С
    return (f(x + h) - f(x - h)) / (h * 2)

# ʵ���ݶ�
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val  # ��ԭֵ
        it.iternext()
        
    return grad

# �ݶȷ�
def gradient_descent(f,init_x,lr,step_num):
    # fΪ�������Ż��ĺ�����init_xΪ��ʼ����
    # lrΪѧϰ��learning rate��step_numΪ�ݶȷ��ظ�����
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr * grad 
    return x

def function1(x):
    return 0.01*x**2 + 0.1*x
def function2(x):
    return x[0]**2 + x[1]**2

