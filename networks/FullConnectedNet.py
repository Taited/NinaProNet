import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self, input_size, class_num):
        super().__init__()
        self.class_num = class_num
        self.input_size = input_size
        self.fcn1 = nn.Sequential(
            nn.Linear(in_features=input_size[0] * input_size[1], out_features=10000),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(10000),
            nn.ReLU())
        self.fcn2 = nn.Sequential(
            nn.Linear(in_features=10000, out_features=1000),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(1000),
            nn.ReLU())
        self.fcn3 = nn.Sequential(
            nn.Linear(in_features=1000, out_features=100),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(100),
            nn.ReLU())
        self.fcn4 = nn.Linear(in_features=100, out_features=self.class_num)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fcn1(x)
        x = self.fcn2(x)
        x = self.fcn3(x)
        x = self.fcn4(x)
        x = F.softmax(x, dim=1)
        return x
