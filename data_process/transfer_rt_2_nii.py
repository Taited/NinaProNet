from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
import os

data_root = '/lyc/LCTSC/dicom'
data_root_list = os.listdir(data_root)
for cur_folder in data_root_list:
    data_path = os.path.join(data_root, cur_folder)
    data_path = os.path.join(data_path, os.listdir(data_path)[0])
    print(data_path)
    for sub_folder in os.listdir(data_path):
        sub_path = os.path.join(data_path, sub_folder)
        if os.path.isdir(sub_path):
            if len(os.listdir(sub_path))==1:
                rt_path = os.path.join(sub_path,os.listdir(sub_path)[0])
            else:
                dicom_path = sub_path
    dcmrtstruct2nii(rt_path, dicom_path, data_path)
