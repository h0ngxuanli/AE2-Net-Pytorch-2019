# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 19:45:49 2019

@author: 18441
"""
import torch
import math
from sklearn.utils import shuffle
from AutoEncoder_Net import AE_Net
from Degration_Net import Dg_Net
from data_loader import get_dataloader1,get_dataloader2
from torch import optim
import numpy as np
from torch import tensor,autograd
#gt needs to be shuffled with X1,X2
def model(X1, X2, y, H, dims, lr, epochs, batch_size, para_lamda=1,activation='sigmoid'):
    #y_ = tensor([i for i in y])
    net_ae1 = AE_Net(1,dims[0],para_lamda,activation)
    net_ae2 = AE_Net(2,dims[1],para_lamda,activation)
    net_dg1 = Dg_Net(1,dims[2],para_lamda,activation)
    net_dg2 = Dg_Net(2,dims[3],para_lamda,activation)######参数顺序不要变！！！！ 
    param1 = list(net_ae1.parameters())+list(net_ae2.parameters())
    param2 = list(net_ae1.parameters())+list(net_ae2.parameters())+list(net_dg1.parameters())+list(net_dg2.parameters())
    param3 = list(net_dg1.parameters())+list(net_dg2.parameters())
   # print(param1)
   # print(param2)
   # print(param3)
    optimizer_pre = optim.Adam([{'params':net_ae1.parameters()},{'params':net_ae2.parameters()}], lr = lr[0])
    optimizer_ae = optim.Adam([{'params':net_ae1.parameters()},{'params':net_ae2.parameters()},{'params':net_dg1.parameters()},{'params':net_dg2.parameters()}], lr = lr[1])
    optimizer_dg = optim.Adam([{'params':net_dg1.parameters()},{'params':net_dg2.parameters()}], lr = lr[2])
    #pretrain the ae_net
   # print(X1)
    for k in range(epochs[0]):
        dataloader_pre = get_dataloader1(X1,X2,y,batch_size)#没有真正的shuffle顺序不变！！！！！
        for i,item in enumerate(dataloader_pre):
            batch_x1,batch_x2,batch_y = item
            #print(batch_x1.is_leaf)
          #  y_[i*batch_size:(i+1)*batch_size,:].data.copy_(batch_y)
            optimizer_pre.zero_grad()
            loss_pretrain = net_ae1.loss_recon(batch_x1) + net_ae2.loss_recon(batch_x2)
            loss_pretrain.backward()
            optimizer_pre.step()
            output = "Pre_epoch : {:.0f}, Batch : {:.0f}  ===> Reconstruction loss = {:.4f} ".format((k + 1), i+1,loss_pretrain)
            print(output)
    #H = tensor(H,requires_grad = True).float()#tensor操作使其不是叶子节点######################
   # print(H.is_leaf) #直接传入H的为叶子节点
  #  print(X1)
   # fea1_latent = net_dg1.get_g(H)
   # fea2_latent = net_dg2.get_g(H)
    #train the dg_net
    num_samples = X1.shape[0]
    num_batchs = math.ceil(num_samples / batch_size)
    X1 = tensor(X1,requires_grad=True)
    X2 = tensor(X2,requires_grad = True)
    y = tensor(y,requires_grad = True)
    for j in range(epochs[1]):
        """
        dataloader_dg = get_dataloader2(X1,X2,y_,H,fea1_latent,fea2_latent,batch_size)#注意参数顺序
        for i,item in enumerate(dataloader_dg):
        """
        #print(type(X1))
      #  if(X1.dtype == np.float32):
           # X1,X2,y_,H,fea1_latent,fea2_latent = tensor(shuffle(X1,X2,y_,H,fea1_latent,fea2_latent),requires_grad=True)
        #else:
        
#        X1,X2,y,H,fea1_latent,fea2_latent = [tensor(i,requires_grad = True) for i in shuffle(X1.detach().numpy(),X2.detach().numpy(),y.detach().numpy(),H.detach().numpy(),fea1_latent.detach().numpy(),fea2_latent.detach().numpy())]
        #print(X1)
        X1,X2,y,H = [tensor(i) for i in shuffle(X1.detach().numpy(),X2.detach().numpy(),y.detach().numpy(),H.detach().numpy())]
        for num_batch_i in range(int(num_batchs) - 1):    
            start_idx, end_idx = num_batch_i * batch_size, (num_batch_i + 1) * batch_size
            end_idx = min(num_samples, end_idx)
         #   print(H[start_idx:end_idx,...])
            batch_x1,batch_x2,batch_y,batch_h = X1[start_idx:end_idx,...],X2[start_idx:end_idx,...],y[start_idx:end_idx,...],H[start_idx:end_idx,...]#注意参数顺序
          #  print(batch_h)
            batch_x1.requires_grad_()
            batch_x2.requires_grad_()
            batch_h.requires_grad_()
            #print(batch_x1.is_leaf)
            #print(batch_x1)
            
            #y_[i*batch_size:(i+1)*batch_size,:].data.copy_(batch_y)
            #print(batch_y)
            #train the ae_net
            batch_g1,batch_g2 = net_dg1.get_g(batch_h),net_dg1.get_g(batch_h)
            with torch.autograd.set_detect_anomaly(True):
                optimizer_ae.zero_grad()
    #            batch_h.retain_grad()
                loss_total = net_ae1.loss_total(batch_x1,batch_g1) + net_ae2.loss_total(batch_x2,batch_g2)
                #print(batch_x2.is_leaf)
                #print('-------')
                #print(param2[0].grad)
                loss_total.backward(retain_graph = True)
                optimizer_ae.step()
                z_half1 = net_ae1.get_z_half(batch_x1)
                z_half2 = net_ae2.get_z_half(batch_x2)
                optimizer_h = optim.Adam([batch_h],lr = lr[3])
                #train the dg_net
                optimizer_dg.zero_grad()
              #  batch_h.retain_grad()
            #  optimizer_h = optim.Adam([batch_h],lr = lr[3])
                loss_degration = net_dg1.loss_dg(z_half1,batch_h) + net_dg2.loss_dg(z_half2,batch_h)
                loss_degration.backward()
            # print(batch_h.is_leaf)
                optimizer_dg.step()
                optimizer_h.step()
                #print(batch_h.is_leaf)
                #optimizer_h.step()
                #batch_h -= lr[3]*batch_h.grad
                #print(batch_h[1:2].data[0])
               # print(batch_h.grad)
            #    print(H[start_idx:end_idx,...])
                #batch_h = batch_h - lr[3]*batch_h.grad
              #  print(H[start_idx:end_idx,...])
              #  print(batch_h)
               # print(batch_h.grad)
                batch_g1_new = net_dg1.get_g(batch_h)
                batch_g2_new = net_dg2.get_g(batch_h)
    #                print(H[i*batch_size:(i+1)*batch_size,:],batch_h[0])
              #  print(batch_h.is_leaf)
                #  h_save = tensor([i for i in batch_h])
                #H.requires_grad = False
                #H[start_idx:end_idx,...]与batch_h连在一起？？？？
               # print(batch_h)
                
                H[start_idx:end_idx,...].data.copy_(batch_h)
                #print(H[start_idx:end_idx,...])
                #a = batch_h.data
               # print(H[start_idx:end_idx,...])
                loss_total = net_ae1.loss_total(batch_x1,batch_g1_new) + net_ae2.loss_total(batch_x2,batch_g2_new)
                output = "Epoch : {:.0f} -- Batch : {:.0f} ===> Total training loss = {:.4f} ".format((j + 1), (num_batch_i + 1),loss_total)
                print(output)
    return H,y
    
            
    