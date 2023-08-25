from torch import tensor
from torch import optim
import numpy as np
import torch
import math
from . import metrics
"""
y = tensor(np.array([1,1,1]))
x = tensor(np.array([[1,2,3],[4,5,6],[7,8,9]]))
x[1:2,].data.copy_(y.data)
print(x.numpy())
"""
"""
from sklearn.utils import shuffle
x = tensor(np.array([[1,2],[5,6],[7.0,9]]))

z = tensor(np.array([[1,2,3],[4,5,6],[7.0,8,9]]))
x,y,z = [tensor(i,requires_grad = True) for i in shuffle(x.numpy(),y.numpy(),z.numpy())]
print(x)
"""
#y = tensor(np.array([[1,2,3],[4,5,6],[7.0,8,9]]))
#y[1:2,...].data.copy_(tensor([1,1,1]).data)
#print(y[1:2])
#print(math.ceil(5/3))
"""
x = tensor([[3,4],[5,6.0]],requires_grad=True)
print(x)
y1 = x + 2
y2 = x-2
def fun(y1,y2):
    return (y1*y2).mean()
#y.retain_grad()
#z = y1 * y2 * 3
out = fun(y1,y2)
out.backward()
print(x.is_leaf)
print(x-0.01*x.grad)
optimizer_h = optim.Adam([y],lr = 0.01)
optimizer_h.step()

print(x.grad,x)
"""
print(metrics.acc([1,1,1,1,1,1],[2,2,2,2,2,2]))