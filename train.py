from __future__ import absolute_import, print_function
import time
import torch.optim as optim
import torch.tensor
import torch.backends.cudnn as cudnn
from torchvision import transforms
from dataloaders.Structseg_dataloader import *
from torch.utils.data import DataLoader
from util.train_test_func import *
from util.parse_config import parse_config
from networks.Unet_Separate_2 import Unet_Separate_2
from networks.Unet_Separate_3 import Unet_Separate_3
from networks.Unet_Separate_3_dis import Unet_Separate_3_dis
from networks.Unet_Separate_3_dis2 import Unet_Separate_3_dis2
from networks.Unet_Separate_4 import Unet_Separate_4
from networks.Unet_Separate_5 import Unet_Separate_5
from networks.Unet_Separate_6 import Unet_Separate_6
from networks.Unet_Separate_7 import Unet_Separate_7
from networks.UnetSE_Separate_3 import UnetSE_Separate_3
from networks.Unet import Unet
from networks.Adapt_transform import Adapt_transform,Adapt_transform2
from networks.Unet_adapt_Separate import Unet_adapt_Separate
from networks.DenseSepUnet import DenseSepUnet
from networks.Unet_Separate import Unet_Separate
from networks.Unet_Res import Unet_Res
from networks.DeepMedic import DeepMedic
from losses.loss_function import TestDiceLoss, AttentionExpDiceLoss, BinaryCrossEntropy
from util.visualization.visualize_loss import dice_visualize
from util.visualization.show_param import show_param
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'Unet':
            return Unet

        if name == 'Unet_Res':
            return Unet_Res

        if name == 'Unet_Separate':
            return Unet_Separate

        if name == 'Unet_Separate_2':
            return Unet_Separate_2

        if name == 'Unet_Separate_3':
            return Unet_Separate_3

        if name == 'Unet_Separate_4':
            return Unet_Separate_4

        if name == 'Unet_Separate_5':
            return Unet_Separate_5

        if name == 'Unet_Separate_6':
            return Unet_Separate_6

        if name == 'Unet_Separate_7':
            return Unet_Separate_7

        if name == 'UnetSE_Separate_3':
            return UnetSE_Separate_3

        if name == 'DenseSepUnet':
            return DenseSepUnet

        if name == 'Unet_Separate_3_dis':
            return Unet_Separate_3_dis

        if name == 'Unet_Separate_3_dis2':
            return Unet_Separate_3_dis2

        if name == 'DeepMedic':
            return DeepMedic

        if name == 'Adapt_transform':
            return Adapt_transform

        if name == 'Adapt_transform2':
            return Adapt_transform2

        if name == 'Unet_adapt_Separate':
            return Unet_adapt_Separate
        # add your own networks here
        print('unsupported network:', name)
        exit()


def train():
    # 1, load configuration parameters
    print('1.Load parameters')
    config = parse_config(config_file)
    config_data = config['data']  # 包含数据的各种信息,如data_shape,batch_size等
    config_net = config['network']  # 网络参数,如net_name,base_feature_name,class_num等
    config_train = config['training']

    train_patch_size = config_data['train_patch_size']
    test_patch_size = config_data['test_patch_size']
    stride_xy = config_data['stride_xy']
    stride_z = config_data['stride_z']
    stride = [stride_z, stride_xy, stride_xy]
    batch_size = config_data.get('batch_size', 4)
    extra_adapt = True
    double_input = False
    net_type = config_net['net_type']
    class_num = config_net['class_num']

    lr = config_train.get('learning_rate', 1e-3)
    best_dice = config_train.get('best_dice', 0.5)
    random_seed = config_train.get('random_seed', 1)
    random.seed(random_seed)  # 给定seed value,决定了后面的伪随机序列
    cudnn.benchmark = True
    cudnn.deterministic = True
    best_dice_iter = 0
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    # 2, load data
    print('2.Load data')
    trainData = StrusegDataloader(config=config_data,
                                  split='train',
                                  transform=transforms.Compose([
                                      ExtractCertainClass(class_wanted=[4]),
                                      CropBound(pad=[16, 32, 32], mode='coarseg'),
                                      RandomCrop(train_patch_size),
                                      ToTensor(doubleinput=double_input),
                                  ]))
    validData = StrusegDataloader(config=config_data,
                                  split='valid',
                                  transform=transforms.Compose([
                                      ExtractCertainClass(class_wanted=[4]),
                                      CropBound(pad=[16, 32, 32], mode='label')
                                  ]),
                                  random_sample=False)

    def worker_init_fn(worker_id):
        random.seed(random_seed + worker_id)

    trainLoader = DataLoaderX(trainData, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    validLoader = DataLoaderX(validData, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # 3. creat model
    print('3.Creat model')
    hu_lis = np.asarray([[-4, 2]])
    norm_lis = np.asarray([[0.5, 0.5]])
    smooth_lis = np.asarray([[2, 1]])
    net_class = NetFactory.create(net_type)
    net = net_class(inc=config_net.get('input_channel', 1),
                    n_classes=class_num,
                    base_chns=config_net.get('base_feature_number', 16),
                    droprate=config_net.get('drop_rate', 0.2),
                    norm='in',
                    depth=config_net.get('depth', False),
                    dilation=config_net.get('dilation', 1),
                    separate_direction='axial',
                    )
    net = torch.nn.DataParallel(net, device_ids=[0, 1]).cuda()
    if extra_adapt:
        adapt_trans = Adapt_transform2(hu_lis=hu_lis, norm_lis=norm_lis, smooth_lis=smooth_lis)
        adapt_trans = torch.nn.DataParallel(adapt_trans, device_ids=[0, 1]).cuda()
    else:
        adapt_trans = None
    if config_train['load_weight']:
        weight = torch.load(config_train['model_path'], map_location=lambda storage, loc: storage)
        net.load_state_dict(weight)
    if config_train['load_transform']:
        if extra_adapt:
            weight = torch.load(config_train['adapt_model_path'], map_location=lambda storage, loc: storage)
            adapt_trans.load_state_dict(weight)

    show_param(net)  # 计算网络总参数量
    dice_eval = TestDiceLoss(n_class=class_num)
    loss_func = AttentionExpDiceLoss(n_class=class_num, alpha=1, weight=[1, 1])
    binar_cross = BinaryCrossEntropy()
    show_loss = dice_visualize(class_num)

    Adamoptimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=config_train.get('decay', 1e-7))
    Adamscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Adamoptimizer, mode='max', factor=0.2, patience=10,
                                                               threshold=0.001)
    if extra_adapt:
        Adapt_optimizer = optim.Adam(adapt_trans.parameters(), lr=lr, weight_decay=config_train.get('decay', 1e-7))
        Adapt_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(Adapt_optimizer, mode='max', factor=0.2,
                                                                     patience=10, threshold=0.001)
    downsample = torch.nn.AvgPool3d(4, 4)


if __name__ == '__main__':
    config_file = str('config/train_szwt.txt')
    assert(os.path.isfile(config_file))
    train(config_file)
