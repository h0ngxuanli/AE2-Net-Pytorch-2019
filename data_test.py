import h5py
from scipy.io import loadmat,savemat
import numpy as np
from collections import Counter
import pandas as pd
#data = h5py.File(r'C:\Users\18441\Desktop\data202.mat',mode = 'r')
data = loadmat(r'C:\Users\18441\Desktop\DGMM_TSCCA_Data.mat')
#print(np.array(data['img_name']))
VBM = data['VBM']
FDG = data['FDG']
AV45 = data['AV45']
label = data['Label']

#label = np.array(label)
#Counter(list(label))
label = [i[0] for i in label]
VBM = pd.DataFrame(VBM)
FDG = pd.DataFrame(FDG)
AV45 = pd.DataFrame(AV45)
label = pd.DataFrame(label)
data = pd.concat([VBM,FDG,AV45,label],axis = 1)
data.columns = list([i for i in range(349)])
data = data.sort_values(by = data.columns[-1])
NC = data.values[:211,:-1]
SMC = data.values[211:211+82,:-1]
EMC = data.values[211+82:211+82+273,:-1]
LMC = data.values[211+82+273:211+82+273+187,:-1]
AD = data.values[211+82+273+187:211+82+273+187+160,:-1]
print(NC.shape,SMC.shape,EMC.shape,LMC.shape,AD.shape)
label = data.values[:,-1].reshape(913,1)
index = np.cumsum([0,211,82,273,187,160])
data1 = {'NC':NC,'SMC':SMC,'EMC':EMC,'LMC':LMC,'AD':AD}
data = dict()
modal = ['VBM','FDG','AV45']
sym = ['NC','SMC','EMC','LMC','AD']
count_m = 0
print(label)
for i in modal:
    count_s = 0
    for j in sym:
        data[i+'_'+j] = np.hsplit(data1[j],3)[count_m]
        data[i+'_'+j+'_'+'label'] = label[index[count_s]:index[count_s+1]] 
        count_s+=1
    count_m+=1
savemat(r'C:\Users\18441\Desktop\3modalsfor5class.mat', data)