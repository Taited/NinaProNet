from dataloaders.NinaPro_dataset import *
import torchvision.transforms as tt
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class NinaProNet(nn.Module):
    def __init__(self, class_num=None, base_features=16):
        super(NinaProNet, self).__init__()
        self.class_num = class_num
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=10,
                            out_channels=base_features * 2,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.BatchNorm1d(base_features * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=base_features * 2,
                            out_channels=base_features * 4,
                            kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(base_features * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=base_features * 4,
                            out_channels=base_features * 4,
                            kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(base_features * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=base_features * 4,
                            out_channels=base_features * 4,
                            kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(base_features * 4),
            nn.ReLU()
        )

        self.mlp1 = torch.nn.Linear(base_features * 8 * 8, 100)
        self.mlp2 = torch.nn.Linear(100, self.class_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.mlp1(x.view(x.size(0), -1))
        x = F.relu(x)
        x = self.mlp2(x)
        x = F.softmax(x)
        return x


class FcnNet(nn.Module):
    def __init__(self, class_num=None, base_features=16):
        super(FcnNet, self).__init__()
        self.class_num = class_num
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=10,
                            out_channels=base_features * 2,
                            kernel_size=3,
                            stride=1,
                            padding=1),
            torch.nn.BatchNorm1d(base_features * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=base_features * 2,
                            out_channels=base_features * 4,
                            kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(base_features * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=base_features * 4,
                            out_channels=base_features * 4,
                            kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(base_features * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=base_features * 4,
                            out_channels=base_features * 4,
                            kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm1d(base_features * 4),
            nn.ReLU()
        )

        self.mlp1 = torch.nn.Linear(base_features * 8 * 8, 100)
        self.mlp2 = torch.nn.Linear(100, self.class_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.mlp1(x.view(x.size(0), -1))
        x = F.relu(x)
        x = self.mlp2(x)
        x = F.softmax(x)
        return x


if __name__ == '__main__':
    root = r'D:\Dataset\NinaproEMG\DB1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 50
    cutoff_frequency = 45
    sampling_frequency = 100
    wn = 2 * cutoff_frequency / sampling_frequency
    myDataset = NinaProDataset(root=root,
                               butterWn=wn,
                               window_length=128,
                               random_sample=1024,
                               transform=tt.Compose([
                                   ToTensor()]))
    net = NinaProNet(myDataset.class_num)
    net = net.to(device)
    trainLoader = DataLoaderX(myDataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    net.train()
    loss_list = []
    for epoch in range(epochs):
        total_loss = 0
        for i, sample in enumerate(trainLoader):
            data, label = sample['data'].to(device), sample['label'].to(device)
            prediction = net(data)
            train_loss = loss_func(prediction, label.squeeze(1).squeeze(1))
            total_loss += train_loss
            optimizer.zero_grad()  # 梯度归零
            train_loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            print(train_loss)
        loss_list.append(train_loss.cpu().detach().numpy()/(i + 1))
    plt.plot(np.array(loss_list))
    plt.show()
