import os
import numpy as np
from data_process.data_process_func import load_nifty_volume_as_array, save_array_as_nifty_volume

data_root = '/lyc/LCTSC/dicom'
data_root_list = os.listdir(data_root)
label_list = ['mask_Lung_L.nii.gz','mask_Lung_R.nii.gz','mask_Heart.nii.gz','mask_Esophagus.nii.gz','mask_SpinalCord.nii.gz']
for cur_folder in data_root_list:
    data_path = os.path.join(data_root, cur_folder)
    data_path = os.path.join(data_path, os.listdir(data_path)[0])
    for i in range(len(label_list)):
        cur_label=load_nifty_volume_as_array(os.path.join(data_path,label_list[i]))
        if i ==0:
            label = np.zeros_like(cur_label)
        label[np.where(cur_label==255)]=i+1
        print(np.max(label))
    save_array_as_nifty_volume(label, data_path+'/label.nii.gz')
    print('Successfully transfer', data_path)
