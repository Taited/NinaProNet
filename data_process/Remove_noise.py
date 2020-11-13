#!/usr/bin/env python
import os
import numpy as np
from data_process.data_process_func import *




def remove_noise(label, label_wanted, threshold=None):
    for i in label_wanted:
        clabel = np.zeros_like(label)
        clabel[np.where(label==i)]=1
        label[np.where(label==i)]=0
        nnlabel = get_largest_two_component(clabel, threshold=threshold[i])
        label += nnlabel*i
    return label

if __name__ == '__main__':
    data_root = '/lyc/LUNG/StructSeg19_task3/Task3_Thoracic_OAR/Thoracic_OAR/'
    filename_list = ['coarseg.nii.gz','coarseg.nii.gz']
    modelist = [ 'valid','train']
    label_lis = [1,2]
    save_as_nifty = True
    for mode in modelist:
        filelist =os.listdir(os.path.join(data_root, mode))
        filenum = len(filelist)
        for ii in range(filenum):
            label_path = os.path.join(data_root, mode, filelist[ii], filename_list[0])
            nlabel_path = os.path.join(data_root, mode, filelist[ii], filename_list[1])
            label = load_nifty_volume_as_array(label_path)
            label = remove_noise(label, label_lis)
            if save_as_nifty:
                save_array_as_nifty_volume(label, nlabel_path)
            print('成功储存', filelist[ii])
            if save_as_nifty:
                save_array_as_nifty_volume(label, nlabel_path)


