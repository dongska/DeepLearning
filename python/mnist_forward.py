# coding: utf-8
import numpy as np
import pickle
from activationFunction import sigmoid_function, softmax_function
from mnist import load_mnist

# 加载现成网络的函数
def init_network():
    with open("python\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

# 三层神经网络的前向传播
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_function(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_function(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax_function(a3)    #作为分类任务，使用softmax函数

    return y

# 加载MNIST数据集
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True,one_hot_label = False)
# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)

# 加载现成网络
network = init_network()
accuracy_cnt = 0

batch_size = 100
# 以捆（批处理）batch预测准确度，一捆100个
batch =  list(range(0,len(x_test),batch_size))
print(batch)
for i in batch:
    x_batch = x_test[i:i+batch_size,:] # 测试数据X,100*784，一捆，共100捆
    #print(x_batch.shape)
    t_batch = t_test[i:i+batch_size]# 测试数据tag,100*1，一捆，共100捆
    #print(t_batch.shape)
    y_batch = predict(network,x_batch)
    # print(y_batch[1])
    p = np.argmax(y_batch,axis = 1) # y_batch:100*10为十个类别的概率，使用np.argmax(axis = 1)找到每个图片概率最大的下标
    # print(p)
    #print(p == t_batch)
    accuracy_cnt += np.sum(p == t_batch)

accuracy = accuracy_cnt / len(x_test)
print(accuracy)