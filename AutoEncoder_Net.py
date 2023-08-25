# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:57:07 2019

@author: 18441
"""

import torch.nn as nn
from torch import tensor
import numpy as np
class AE_Net(nn.Module):
    def __init__(self,v,dim_encoder,lamda,activation):
        super(AE_Net,self).__init__()
        self.v = v
        self.dim_encoder = dim_encoder
        self.dim_decoder = [i for i in reversed(dim_encoder)]
        self.lamda = lamda
        self.activation = ({'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'tanh':nn.Tanh()}).get(activation)
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        for i in range(1,len(self.dim_encoder)):
            self.encoder.extend([nn.Linear(self.dim_encoder[i-1],self.dim_encoder[i]),self.activation])
            self.decoder.extend([nn.Linear(self.dim_decoder[i-1],self.dim_decoder[i]),self.activation])
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)
        self.weights = self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        return
        """
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        """
    def get_z(self,x):
        z_half = self.encoder(x)
        z = self.decoder(z_half)
        return z
    def get_z_half(self,x):
        z_half = self.encoder(x)
        return z_half
    def loss_recon(self,x):
        loss_recon = nn.MSELoss()
        z = self.get_z(x)
        return loss_recon(x,z)
    def loss_total(self,x,g):
        loss_deg = nn.MSELoss()
        loss_reconstruct = nn.MSELoss()
        z_half = self.encoder(x)
        z = self.decoder(z_half)
        loss= 0.5*loss_reconstruct(x,z)+self.lamda*0.5*loss_deg(z_half,g)
        return loss
        
#x = AE_Net(1,[3,2,1],1,'sigmoid')
#out = x(tensor(np.linspace(1,6,6).reshape(2,3)))
#y = tensor([[3,1,1]])
#print(x.parameters)
#for i in x.modules():
#    if type(i)==nn.Linear:
#        print(i.weight,i.bias)