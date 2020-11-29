import torch.nn as nn
import torch.nn.functional as F


class NinaProNet(nn.Module):
    def __init__(self, class_num=None, base_features=16, window_length=256):
        super(NinaProNet, self).__init__()
        self.class_num = class_num
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=10,
                      out_channels=base_features * 2,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm1d(base_features * 2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=base_features * 2,
                      out_channels=base_features * 4,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(base_features * 4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=base_features * 4,
                      out_channels=base_features * 4,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(base_features * 4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=base_features * 4,
                      out_channels=base_features * 4,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(base_features * 4),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.mlp1 = nn.Sequential(
            nn.Linear(base_features * 4 * int(window_length / 8), 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.mlp2 = nn.Linear(100, self.class_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        x = F.softmax(x, dim=1)
        return x
