from util.visualization.visualize_loss import loss_visualize
from prefetch_generator import BackgroundGenerator
from dataloaders.NinaPro_dataset import *
from util.parse_config import parse_config
from torch.utils.data import DataLoader
from networks.NetFactory import NetFactory
import torchvision.transforms as tt
import torch
import time


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class AccuracyMetrics:
    def __init__(self, class_num=None):
        self.shot = np.zeros(class_num)
        self.total = np.zeros(class_num) + 1e-5  # avoid divide zero

    def get_data(self, prediction, label):
        _, predicted_label = torch.max(prediction, dim=1)
        for i in range(len(predicted_label)):
            index = label[i].cpu()
            self.total[index] += 1
            if predicted_label[i] == index:
                self.shot[index] += 1

    def get_prediction(self, prediction, label):
        for i in range(prediction.shape[0]):
            index = int(label[i])
            self.total[index] += 1
            if prediction[i] == index:
                self.shot[index] += 1

    def evaluate(self):
        class_accuracy = self.shot / self.total
        mean_accuracy = np.mean(class_accuracy)
        return mean_accuracy, class_accuracy

    def clean(self):
        self.shot *= 0
        self.total *= 0
        self.total += 1e-5


def train(config):
    # load data config
    config_data = config['data']
    cutoff_frequency = config_data['cutoff_frequency']
    sampling_frequency = config_data['sampling_frequency']
    # wn = 2 * cutoff_frequency / sampling_frequency
    wn = None
    window_length = config_data['window_length']
    iter_num = config_data['iter_num']
    train_batch_size = config_data['train_batch_size']
    root = config_data['data_root']

    # load net config
    config_net = config['network']
    net_name = config_net['net_type']

    # load train config
    config_train = config['training']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = config_train['maximal_epoch']
    print_step = config_train['print_step']

    # initiate dataset
    train_dataset = NinaProDataset(root=root,
                                   split='train',
                                   butterWn=wn,
                                   window_length=window_length,
                                   random_sample=iter_num,
                                   transform=tt.Compose([FeatureExtractor()]))
    valid_dataset = NinaProDataset(root=root,
                                   split='valid',
                                   butterWn=wn,
                                   window_length=window_length,
                                   transform=tt.Compose([FeatureExtractor()]))
    trainLoader = DataLoaderX(train_dataset, batch_size=train_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    validLoader = DataLoaderX(valid_dataset, batch_size=len(valid_dataset), shuffle=False, num_workers=6, pin_memory=True)

    # initiate net
    net = NetFactory.create(net_name)

    if config_train['load_weight']:
        weight = torch.load(config_train['model_path'], map_location=lambda storage, loc: storage)
        net.load_state_dict(weight)

    # initiate metrics and loss func
    evaluator = AccuracyMetrics(train_dataset.class_num)

    # train begin
    for epoch in range(epochs):
        evaluator.clean()
        for i, sample in enumerate(trainLoader):
            data, label = sample['data'], sample['label']
            net.fit(data, label)
            prediction = net.predict(data)
            evaluator.get_prediction(prediction, label.squeeze(1))
        train_accuracy, _ = evaluator.evaluate()
        print('{}  train accuracy: {}'.
              format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), train_accuracy))

        # valid
        evaluator.clean()
        for i, sample in enumerate(validLoader):
            data, label = sample['data'], sample['label']
            prediction = net.predict(data)
            evaluator.get_prediction(prediction, label.squeeze(1))
        valid_accuracy, _ = evaluator.evaluate()
        print('{} valid accuracy: {}'.
              format(time.strftime("%Y-%m-%d %H:%M:%S"), valid_accuracy))


if __name__ == '__main__':
    config_file = 'config/train_ml.txt'
    cfg = parse_config(config_file)
    train(config=cfg)
