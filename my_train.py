from prefetch_generator import BackgroundGenerator
from dataloaders.NinaPro_dataset import *
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.autograd import Variable
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

        self.mlp1 = torch.nn.Linear(base_features * 8 * 16, 100)
        self.mlp2 = torch.nn.Linear(100, self.class_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.mlp1(x.view(x.size(0), -1))
        x = F.relu(x)
        x = self.mlp2(x)
        x = F.softmax(x, dim=1)
        return x


class AccuracyMetrics:
    def __init__(self, class_num=None):
        self.shot = np.zeros(class_num)
        self.total = np.zeros(class_num) + 1e-5  # avoid divide zero

    def get_data(self, prediction, label):
        _, predicted_label = torch.max(prediction, dim=1)
        for i in range(len(predicted_label)):
            id = label[i].cpu()
            self.total[id] += 1
            if predicted_label[i] == id:
                self.shot[id] += 1

    def evaluate(self):
        class_accuracy = self.shot / self.total
        mean_accuracy = np.mean(class_accuracy)
        return mean_accuracy, class_accuracy


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=13, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, (float, int)):    #仅仅设置第一类别的权重
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)
        if isinstance(alpha, list):  #全部权重自己设置
            self.alpha = torch.Tensor(alpha)
        self.gamma = gamma

    def forward(self, inputs, targets):
        alpha = self.alpha
        N = inputs.size(0)
        C = inputs.size(1)
        P = inputs
        # ---------one hot start--------------#
        class_mask = inputs.data.new(N, C).fill_(0)  # 生成和input一样shape的tensor
        class_mask = class_mask.requires_grad_()  # 需要更新， 所以加入梯度计算
        ids = targets.view(-1, 1)  # 取得目标的索引
        class_mask.data.scatter_(1, ids.data, 1.)  # 利用scatter将索引丢给mask
        # ---------one hot end-------------------#
        probs = (P * class_mask).sum(1).view(-1, 1)
        print('留下targets的概率（1的部分），0的部分消除\n', probs)
        # 将softmax * one_hot 格式，0的部分被消除 留下1的概率， shape = (5, 1), 5就是每个target的概率

        log_p = probs.log()
        print('取得对数\n', log_p)
        # 取得对数
        loss = torch.pow((1 - probs), self.gamma) * log_p
        batch_loss = -alpha * loss.t()  # 對應下面公式
        print('每一个batch的loss\n', batch_loss)
        # batch_loss就是取每一个batch的loss值

        # 最终将每一个batch的loss加总后平均
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        print('loss值为\n', loss)
        return loss


if __name__ == '__main__':
    root = r'E:\Datasets\NinaproEMG\DB1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 50
    print_step = 10
    cutoff_frequency = 45
    sampling_frequency = 100
    wn = 2 * cutoff_frequency / sampling_frequency
    train_dataset = NinaProDataset(root=root,
                                   split='train',
                                   butterWn=wn,
                                   window_length=256,
                                   random_sample=3200,
                                   transform=tt.Compose([ToTensor()]))
    valid_dataset = NinaProDataset(root=root,
                                   split='valid',
                                   butterWn=wn,
                                   window_length=256,
                                   transform=tt.Compose([ToTensor()]))
    trainLoader = DataLoaderX(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    validLoader = DataLoaderX(valid_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    net = NinaProNet(train_dataset.class_num)
    net = net.to(device)

    evaluator = AccuracyMetrics(train_dataset.class_num)
    loss_func = focal_loss(num_classes=train_dataset.class_num)
    optimizer = optim.Adam(net.parameters())

    net.train()
    train_loss_list = []
    valid_loss_list = []
    for epoch in range(epochs):
        batch_loss = 0
        for i, sample in enumerate(trainLoader):
            data, label = sample['data'].to(device), sample['label'].to(device)
            prediction = net(data)
            train_loss = loss_func(prediction, label.squeeze(1).squeeze(1))
            batch_loss += train_loss
            optimizer.zero_grad()  # 梯度归零
            train_loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            if i % print_step == 0:
                print('\tbatch id: {} train_loss: {}'.format(i, train_loss.cpu().detach().numpy()))
        batch_loss = batch_loss.cpu().detach().numpy() / (i + 1)
        print('epoch: {} train batch loss: {}'.format(epoch, batch_loss))
        train_loss_list.append(batch_loss)

        # 测试
        if epoch % print_step == 0:
            with torch.no_grad():
                batch_loss = 0
                for i, sample in enumerate(validLoader):
                    data, label = sample['data'].to(device), sample['label'].to(device)
                    prediction = net(data)
                    evaluator.get_data(prediction, label.squeeze(1).squeeze(1))
                    valid_loss = loss_func(prediction, label.squeeze(1).squeeze(1))
                    batch_loss += valid_loss
                batch_loss = batch_loss.cpu().detach().numpy() / (i + 1)
                mean_accuracy, _ = evaluator.evaluate()
                valid_loss_list.append(batch_loss)
                print('epoch: {} valid loss: {} mean accuracy: {}'.format(epoch, batch_loss, mean_accuracy))
    plt.figure()
    plt.plot(np.array(train_loss_list), label='train loss')
    plt.plot(np.array(train_loss_list), label='valid loss')
    plt.show()

    mean_accuracy, class_accuracy = evaluator.evaluate()
    plt.figure()
    plt.bar(x=np.arange(0, valid_dataset.class_num),
            height=class_accuracy, label='mean accuracy: {}'.format(mean_accuracy))
    plt.show()
