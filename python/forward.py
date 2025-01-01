# -*- coding: utf-8 -*-
import numpy as np

# 实现sigmoid函数
def sigmoid_function(x):
    y = 1/(1 + np.exp(-x))
    return y

# 三层网络初始化
def init_network():
    network = {}
    network["w1"] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network["b1"] = np.array([0.1,0.2,0.3])
    network["w2"] = np.array([[0.1,0.3],[0.2,0.4],[0.3,0.6]])
    network["b2"] = np.array([0.1,0.2])
    network["w3"] = np.array([[0.1,0.3],[0.2,0.4]])
    network["b3"] = np.array([0.1,0.2])
    return network

def forward(network,x):
    w1,w2,w3 = network["w1"],network["w2"],network["w3"]
    b1,b2,b3 = network["b1"],network["b2"],network["b3"]

    a1 = np.dot(x,w1) + b1
    z1 = sigmoid_function(a1)

    a2 = np.dot(z1,w2) + b2
    z2 = sigmoid_function(a2)

    a3 = np.dot(z2,w3)

    y = a3  # 输出层使用直出激活函数
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network,x)
print(y)

