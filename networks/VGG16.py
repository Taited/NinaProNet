import torch.nn as nn


class SE_VGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # define an empty for Conv_ReLU_MaxPool
        net = []

        # block 1
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        # block 2
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=128, out_channels=128, padding=1, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        # block 3
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, padding=1, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        # block 4
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, padding=1, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        # block 5
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        # add net into class property
        self.extract_feature = nn.Sequential(*net)

        # define an empty container for Linear operations
        classifier = []
        classifier.append(nn.Linear(in_features=512*7*7, out_features=4096))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=4096, out_features=4096))
        classifier.append(nn.ReLU())
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Linear(in_features=4096, out_features=self.num_classes))

        # add classifier into class property
        self.classifier = nn.Sequential(*classifier)


    def forward(self, x):
        feature = self.extract_feature(x)
        feature = feature.view(x.size(0), -1)
        classify_result = self.classifier(feature)
        return classify_result