import numpy as np
def AND(x1,x2):
    w = np.array([0.5,0.5])
    theta = -0.7
    x = np.array([x1,x2])
    if np.sum(w * x) + theta > 0:
        return 1
    else:
        return 0

def OR(x1,x2):
    w = np.array([0.5,0.5])
    theta = -0.3
    x = np.array([x1,x2])
    if np.sum(w * x) + theta > 0:
        return 1
    else:
        return 0
    
def NAND(x1,x2):
    w = np.array([-0.5,-0.5])
    theta = 0.7
    x = np.array([x1,x2])
    if np.sum(w * x) + theta > 0:
        return 1
    else:
        return 0
print(OR(0,0))
