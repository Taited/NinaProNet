# -*- coding:utf-8 -*-
import torch.utils.data as data
import torch
import os
import os.path
import numpy as np


class Segment_data(data.Dataset):
  """
  用于分割的数据集
  """

  def __init__(self, root):
    self.root = root
    self.file_path = os.listdir(root)
    self.file_num = len(self.file_path)

  def __getitem__(self, idx):

    file_path = self.file_path
    file = np.load(self.root +'/' +file_path[idx])
    img = np.atleast_3d(file[0]).transpose(2, 0, 1).astype(np.float64)  # 默认第1张是原图
    img = torch.from_numpy(img).float()
    gt = file[1:] # 后面是各类mask
    gt = torch.from_numpy(gt).float()
    return img, gt

  def __len__(self):
    return self.file_num