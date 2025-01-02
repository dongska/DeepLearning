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

# ʵ��ReLU����
def relu_function(x):
    return np.maximum(0,x)

# ʵ��softMax���� + ��ֹ���
def softmax_function(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid_function(x)
y3 = relu_function(x)
y4 = softmax_function(x)
plt.plot(x, y1,linestyle="--",label="step")
plt.plot(x, y2,linestyle="-",label="sigmoid")
plt.plot(x, y3,linestyle="-",label="relu")
plt.plot(x, y4,linestyle="-",label="softmax")
plt.legend()
plt.ylim(0,1.2)
plt.show()