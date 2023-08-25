# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 06:18:49 2019

@author: 18441
"""
from data_loader import Load_Data
import numpy as np
from train_model import model
from utils.print_result import print_result
from torch import tensor
#if __name__ == 'main':
X1,X2,y = Load_Data(r'C:\Users\18441\Desktop','\\3modalsfor5class.mat')
#X1,X2,y = Load_Data(r'C:\Users\18441\Desktop\AE','\\handwritten_2views.mat')
#print(X1.shape,y.shape)
dim_latent = 100;para_lambda = 1
dims_ae1 = [X1.shape[1],100]#200   
dims_ae2 = [X2.shape[1],100]
dims_dg1 = [dim_latent, 100]
dims_dg2 = [dim_latent, 100]
#batch_size = 50;epochs_pre = 20;epochs_total = 50
batch_size = 50;epochs_pre = 10;epochs_total = 20;epochs_h = 50
#    lr_pre = 1.0e-2;lr_ae = 1.0e-2;lr_dg = 1.0e-3;lr_h = 1.0e-5
lr_pre = 1.0e-3;lr_ae = 1.0e-3;lr_dg = 1.0e-3;lr_h = 1.0e-1
dims = [dims_ae1, dims_ae2, dims_dg1, dims_dg2]
lr = [lr_pre, lr_ae, lr_dg, lr_h]
epochs = [epochs_pre, epochs_total, epochs_h]
#n_clusters = len(set(list(y)))
n_clusters = 2
H = tensor(np.random.uniform(0,1,[X1.shape[0], dims_dg1[0]]),requires_grad = True).float()
H,y = model(X1, X2, y, H, dims, lr, epochs, batch_size, para_lamda=1)
print(H)
#H,y_ = H.detach().numpy(),y_.detach().numpy()
print_result(n_clusters, H, y)
#    accu.append(acc)

