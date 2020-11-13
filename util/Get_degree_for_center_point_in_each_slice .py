import GeodisTK
from scipy import ndimage
import os
# from scipy import ndimage
from data_process.data_process_func import load_nifty_volume_as_array, save_array_as_nifty_volume
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from skimage import morphology


def geodesic_distance(img, seed, distance_threshold):
    threshold = distance_threshold
    if (seed.sum() > 0):
        geo_dis = GeodisTK.geodesic3d_raster_scan(img, seed, 1.0, 2)
        geo_dis[np.where(geo_dis > threshold)] = threshold
        dis = 1 - geo_dis / threshold  # recale to 0-1
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
    M = cv2.moments(contours[0])  # 计算第一条轮廓的各阶矩，字典模式
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    return center_x, center_y


def get_sin(img):
    contours, cnt = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours).squeeze()  # contours is a n*2 array
    # init as the index of x and y. accelerate calculation
    point_index = np.array([np.zeros_like(img), np.zeros_like(img)]).astype(float)
    for x in range(point_index.shape[1]):
        point_index[0, x, :] = x
    for y in range(point_index.shape[2]):
        point_index[1, :, y] = y
    if contours is not None:
        point_dis = np.zeros([contours.shape[0], img.shape[0], img.shape[1]])
        point_sin = np.zeros([contours.shape[0], img.shape[0], img.shape[1]])
        for contours_id, (contours_x, contours_y) in enumerate(contours):
            x = contours_x - point_index[0]
            y = contours_y - point_index[1]
            point_dis[contours_id] = np.sqrt(np.square(x) + np.square(y))
            ems = 1e-4
            point_sin[contours_id] = (y + ems) / (point_dis[contours_id] + ems)

    min_id = np.argmin(point_dis, axis=0)
    min_sin = np.zeros_like(img).astype(float)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            min_sin[x, y] = point_sin[min_id[x, y], x, y]
    margin = 128 + 80
    plt.imshow(min_sin[margin+10:-margin, margin+30:-margin])
    plt.show()
    return min_sin


def get_contours(img, is_show=False):
    kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (3, 3))
    eroded_img = cv2.erode(img, kernel)
    contour = img - eroded_img
    if is_show:
        margin = 200
        _, plots = plt.subplots(1, 3)
        plots[0].imshow(img[margin:-margin, margin:-margin])
        plots[1].imshow(eroded_img[margin:-margin, margin:-margin])
        plots[2].imshow(contour[margin:-margin, margin:-margin])
        plt.show()
    return contour


def get_vector3d(img):
    MAX_DISTANCE = 50
    point_index = np.array([np.zeros_like(img), np.zeros_like(img), np.zeros_like(img)]).astype(float)
    for z in range(point_index.shape[1]):
        point_index[0, z, :, :] = z
    for x in range(point_index.shape[2]):
        point_index[1, :, x, :] = x
    for y in range(point_index.shape[3]):
        point_index[2, :, :, y] = y

    distance_map = np.zeros([3, img.shape[0], img.shape[1], img.shape[2]]).astype(float)
    margin_z = 3
    for id_z in range(margin_z, img.shape[0]-margin_z):
        img_slice = img[id_z-margin_z:id_z+margin_z]
        img_slice = np.nonzero(img_slice)
        if img_slice[0].shape[0] == 0:
            distance_map[:, id_z-margin_z:id_z+margin_z, :, :] = MAX_DISTANCE
        else:
            for contours_x, contours_y in img_slice:
                # print(_)
                print(contours_x)
                print(contours_y)


def get_vector2d(img):
    MAX_DISTANCE = 50
    index_x = np.zeros([img.shape[1], img.shape[2]]).astype(float)
    index_y = np.zeros([img.shape[1], img.shape[2]]).astype(float)
    for x in range(img.shape[1]):
        index_x[x, :] = x
    for y in range(img.shape[2]):
        index_y[:, y] = y

    vector = np.zeros([2, img.shape[0], img.shape[1], img.shape[2]]).astype(float)

    for id_z in range(img.shape[0]):
        if np.sum(img[id_z]) == 0:
            vector[:, id_z, :, :] = 0  # MAX_DISTANCE
        else:
            contours = np.nonzero(img[id_z])
            point_dis = np.zeros([contours[0].shape[0], img.shape[1], img.shape[2]])
            for contours_id in range(contours[0].shape[0]):
                contours_x, contours_y = contours[0][contours_id], contours[1][contours_id]
                x = contours_x - index_x
                y = contours_y - index_y
                point_dis[contours_id] = np.sqrt(np.square(x) + np.square(y))
            min_id = np.argmin(point_dis, axis=0)
            for x in range(img.shape[1]):
                for y in range(img.shape[2]):
                    vector[0, id_z, x, y] = contours[0][min_id[x, y]] - index_x[x, y]
                    vector[1, id_z, x, y] = contours[1][min_id[x, y]] - index_y[x, y]
    vector[np.where(vector >= MAX_DISTANCE)] = MAX_DISTANCE
    return vector


def euclidean_distance(seed, dis_threshold, spacing=[5, 1, 1]):
    threshold = dis_threshold
    if (seed.sum() > 0):
        euc_dis = ndimage.distance_transform_edt(seed == 0, sampling=spacing)
        euc_dis[euc_dis > threshold] = threshold
        dis = 1 - euc_dis / threshold
    else:
        dis = np.zeros_like(seed, np.float32)
    return dis


'''
读取label，根据label在每张slice中心，生成距离图
'''
if __name__ == '__main__':

    data_root = r'D:\Codes\CV\AdaptSeg\test_delete_after'
    mode = ['train', 'valid']
    show = True  # 是否可视化康康
    geo = False
    blur = True
    degree_savename = 'szwt_test_vector_{}.nii.gz'  # 存储距离图的名称
    for submode in mode:
        for item in os.listdir(os.path.join(data_root, submode)):
            item_path = os.path.join(data_root, submode, item)
            label = load_nifty_volume_as_array(os.path.join(item_path, 'label.nii.gz'))
            mask = np.where(label == 1)
            nlabel = np.zeros_like(label)
            center = np.zeros([label.shape[0], 2])
            nlabel[mask] = 1
            print(item, np.sum(nlabel))

            start = time.process_time()
            image = load_nifty_volume_as_array(os.path.join(item_path, 'data.nii.gz'))
            contours = np.zeros_like(nlabel).astype(float)
            for i in range(nlabel.shape[0]):
                contours[i] = get_contours(nlabel[i])
            vector = get_vector2d(contours)
            end = time.process_time()
            print('Running time: %s Seconds'%(end-start))

            margin = 128 + 80
            if show:
                f, plots = plt.subplots(1, 3)
                plots[0].imshow(label[-36, margin:-margin, margin:-margin])
                plots[1].imshow(vector[0, -36, margin:-margin, margin:-margin])
                plots[2].imshow(vector[1, -36, margin:-margin, margin:-margin])
                plt.show()
            save_array_as_nifty_volume(vector[0], os.path.join(item_path, degree_savename.format('x')))
            save_array_as_nifty_volume(vector[1], os.path.join(item_path, degree_savename.format('y')))
            stop = 1
