# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 18:49:56 2019

@author: 18441
"""
from torch import tensor,autograd
import torch.nn as nn
import numpy as np
class Dg_Net(nn.Module):
    def __init__(self,v,dim_dg,lamda,activation):
        super(Dg_Net,self).__init__()
        self.v = v
        self.dim_dg = dim_dg
        self.lamda = lamda
        self.activation = {'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'tanh':nn.Tanh()}.get(activation)
        self.degration = nn.ModuleList([])
        for i in range(1,len(self.dim_dg)):
            self.degration.extend([nn.Linear(self.dim_dg[i-1],self.dim_dg[i]),self.activation])
        self.degration = nn.Sequential(*self.degration)
        self.weights  = self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        return
        """
    def forward(self,h):
        g = self.degration(h)
        return g
        """
    def get_g(self,h):
        #print(h.is_leaf)
        g = self.degration(h)
        return g
    def loss_dg(self,z_half,h):#注意传参顺序
        g = self.degration(h)
        loss_deg = nn.MSELoss()
#        g = tensor(g,requires_grad = True)
#       z_half = tensor(z_half,requires_grad = True)
        loss = 0.5*loss_deg(z_half,g)
        return loss
#    def get_gradient_H(self,h):
#        gradient = h.grad
#        return gradient
#x = Dg_Net(1,[50,100],1,'sigmoid')
#       
""" 
from test import fun
x = tensor([[3,4],[5,6.0]],requires_grad=True)
y1 = x + 2
y2 = x-2        
out = fun(y1,y2)
out.backward()
print(x.is_leaf)
"""
        
        