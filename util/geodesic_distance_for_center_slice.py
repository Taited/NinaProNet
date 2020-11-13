import GeodisTK
from scipy import  ndimage
import os
#from scipy import ndimage
from data_process.data_process_func import load_nifty_volume_as_array,save_array_as_nifty_volume
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import morphology
def geodesic_distance(img, seed, distance_threshold):
    threshold = distance_threshold
    if(seed.sum() > 0):
        geo_dis = GeodisTK.geodesic3d_raster_scan(img, seed, 1.0, 2)
        geo_dis[np.where(geo_dis > threshold)] = threshold
        dis = 1-geo_dis/threshold  # recale to 0-1
    else:
        dis = np.zeros_like(img, np.float32)
    return dis

def get_center_cor(img):
    '''
    get 2d binary img center corardiate
    :param img: 2d binary img
    :return:
    '''
    contours, cnt = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0]) # 计算第一条轮廓的各阶矩，字典模式
    center_x = int(M["m10"]/M["m00"])
    center_y = int(M["m01"]/M["m00"])
    return center_x, center_y

def euclidean_distance(seed, dis_threshold, spacing=[5, 1, 1]):
    threshold = dis_threshold
    if(seed.sum() > 0):
        euc_dis = ndimage.distance_transform_edt(seed==0, sampling=spacing)
        euc_dis[euc_dis > threshold] = threshold
        dis = 1-euc_dis/threshold
    else:
        dis = np.zeros_like(seed, np.float32)
    return dis


if  __name__ == '__main__':
    data_root = '/home/uestc-c1501c/StructSeg/Lung_GTV_n/'
    mode = ['train', 'valid']
    show = False
    geo = True
    save = True
    for submode in mode:
        for item in os.listdir(os.path.join(data_root, submode)):
            item_path = os.path.join(data_root, submode, item)
            label = load_nifty_volume_as_array(os.path.join(item_path, 'crop_label.nii.gz')).astype(np.uint8)
            center_label = load_nifty_volume_as_array(os.path.join(item_path, 'crop_center_label.nii.gz')).astype(np.uint8)
            image = load_nifty_volume_as_array(os.path.join(item_path, 'crop_data.nii.gz')).astype(np.float32)

            if geo:
                dis = geodesic_distance(image, center_label, 300)
            else:
                dis = euclidean_distance(center_label, 30, [5,1,1])
            if show:
                f, plots = plt.subplots(1, 4)
                plots[0].imshow(center_label[-10])
                plots[1].imshow(label[-10])
                plots[2].imshow(image[-10])
                plots[3].imshow(dis[-10])
                plt.show()
            if save:
                save_array_as_nifty_volume(dis, os.path.join(item_path, 'crop_center_geo_dis.nii.gz'))