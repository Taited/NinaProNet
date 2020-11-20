from dataloaders.NinaPro_dataset import *
import torchvision.transforms as tt
import numpy as np
import torch


if __name__ == '__main__':
    root = r'D:\Dataset\NinaproEMG'
    myDataset = NinaPro_dataset(root,
                                transform=tt.Compose([
                                    ToTensor(),
                                    Resize([1, 400, 10])]))
    for i, sample in enumerate(myDataset):
        print('qwq')
