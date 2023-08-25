# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 22:38:55 2019

@author: 18441
"""
import torch
from scipy.io import loadmat
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
class subDataset1(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self,X1,X2,y):
        self.X1 = X1
        self.X2 = X2
        self.y = y
    #返回数据集大小
    def __len__(self):
        return len(self.X1)
    #得到数据内容和标签
    def __getitem__(self, index):
        data1 = torch.Tensor(self.X1[index]).float()
        data2 = torch.Tensor(self.X2[index]).float()
       # print(data2.type)
        #print(self.y[index])
        data3 = torch.Tensor(self.y[index])
        return data1,data2,data3
class subDataset2(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self,X1,X2,y,h,fea_g1,fea_g2):
        self.X1 = X1
        self.X2 = X2
        self.y = y
        self.h = h
        self.fea_g1 = fea_g1
        self.fea_g2 = fea_g2
    #返回数据集大小
    def __len__(self):
        return len(self.X1)
    #得到数据内容和标签
    def __getitem__(self, index):
        data1 = torch.Tensor(self.X1[index]).float()
        data2 = torch.Tensor(self.X2[index]).float()
        data3 = torch.Tensor(self.y[index])
        data4 = torch.Tensor(self.h[index]).float()
        data5 = torch.Tensor(self.fea_g1[index]).float()
        data6 = torch.Tensor(self.fea_g2[index]).float()
        return data1,data2,data3,data4,data5,data6
def Normalize_Data(x,min=0):
    if min==0:
        scaler=MinMaxScaler([0,1])
    else :# min=-1
        scaler=MinMaxScaler((-1,1))
    norm_x=scaler.fit_transform(x)
    return norm_x
    """
def Load_Data(root,name):
    dataset = h5py.File(root+name,mode = 'r')
    X1,X2,y = dataset['x1'],dataset['x2'],dataset['gt']
    X1,X2,y = X1[()].astype(np.float32),X2[()].astype(np.float32),y[()].astype(np.float32)
   # y = np.array(y,dtype= np.float64)
    X1,X2,y=X1.transpose(),X2.transpose(),y.transpose()
    X1,X2 = Normalize_Data(X1),Normalize_Data(X2)
    return X1,X2,y
    """
def Load_Data(root,name):
    #dataset = h5py.File(root+name,mode = 'r')
    dataset = loadmat(root+name)
   # {'VBM': VBM,'FDG': FDG,'AV45': AV45,'label':label}
    #X1,X2,X3,y = dataset['VBM'],dataset['FDG'],dataset['AV45'],dataset['label']
    #VBM_NC,VBM_AD,FDG_NC,FDG_AD = dataset['VBM_NC'],dataset['VBM_AD'],dataset['FDG_NC'],dataset['FDG_AD']
    #VBM_NC_label,VBM_AD_label = dataset["VBM_NC_label"],dataset["VBM_AD_label"]
 #   print(y)
    VBM_EMC,VBM_AD,FDG_EMC,FDG_AD = dataset['VBM_EMC'],dataset['VBM_AD'],dataset['FDG_EMC'],dataset['FDG_AD']
    VBM_EMC_label,VBM_AD_label = dataset["VBM_EMC_label"],dataset["VBM_AD_label"]
    #print((VBM_NC_label,VBM_AD_label))
    #X1,X2,X3,y = X1[()].astype(np.float32),X2[()].astype(np.float32),X3[()].astype(np.float32),y[()].astype(np.float32)
    y = np.vstack((VBM_EMC_label,VBM_AD_label))
   # y = np.array(y,dtype= np.float64)
   # X1,X2,y=X1.transpose(),X2.transpose(),y.transpose()
    X1 = np.vstack((VBM_EMC,VBM_AD))
    X2 = np.vstack((FDG_EMC,FDG_AD))
    X1,X2,y = X1[()].astype(np.float32),X2[()].astype(np.float32),y[()].astype(np.float32)
    X1,X2 = Normalize_Data(X1),Normalize_Data(X2)
   # X1,X2,X3 = Normalize_Data(X1),Normalize_Data(X2),Normalize_Data(X3)
    return X1,X2,y
def get_dataloader1(X1,X2,y,batch_size):
    dataset = subDataset1(X1,X2,y)
    Data_Loader = DataLoader.DataLoader(dataset,batch_size= batch_size, shuffle = True)#num_workers= 4
    return Data_Loader
def get_dataloader2(X1,X2,y,h,fea_g1,fea_g2,batch_size):
    dataset = subDataset2(X1,X2,y,h,fea_g1,fea_g2)
    Data_Loader = DataLoader.DataLoader(dataset,batch_size= batch_size, shuffle = True)#num_workers= 4
    return Data_Loader    
X1,X2,y = Load_Data(r'C:\Users\18441\Desktop','\\3modalsfor5class.mat')
print(y)
