# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:23:27 2025

@author: exx
"""
import os
import numpy as np
dir_path = os.getcwd()
path='\\spreadcapt'
root=dir_path+path
os.chdir(root+'\\train_data')
files=os.listdir()
a2=np.array([])
for file in files:
    a1=np.load(dir_path+path+'\\train_data'+'\\'+file, allow_pickle=True)['data']
    try:
        a2 =np.concatenate((a1, a2),1)
    except:
        a2=a1
os.chdir(dir_path)
np.savez('training_dataset_spreadcapt.npz', data=np.array(a2, dtype=object))
os.chdir(root+'\\test_data')
files=os.listdir()
a2=np.array([])
for file in files:
    a1=np.load(dir_path+path+'\\test_data'+'\\'+file, allow_pickle=True)['data']
    try:
        a2 =np.concatenate((a1, a2),1)
    except:
        a2=a1
os.chdir(dir_path)
np.savez('testing_dataset_spreadcapt.npz', data=np.array(a2, dtype=object))
