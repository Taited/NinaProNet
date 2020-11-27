# -*- coding: utf-8 -*-
import numpy as np
from visdom import Visdom


class dice_visualize(object):
    def __init__(self, class_num, env='dice'):
        self.viz = Visdom(env=env)
        epoch = 0
        self.dice = self.viz.line(X=np.array([epoch]),
                                  Y=np.zeros([1, class_num+1]),  # +1是因为除去背景还有train与test
                                  opts=dict(showlegend=True))

    def plot_dice(self, epoch, epoch_dice):
        train_dice_mean = np.asarray([epoch_dice['train_dice'].mean(axis=0)])
        valid_dice_classes = epoch_dice['valid_dice']
        valid_dice_mean = np.asarray([valid_dice_classes.mean(axis=0)])
        dice = np.concatenate((train_dice_mean,valid_dice_mean, valid_dice_classes), axis=0)[np.newaxis, :]
        self.viz.line(
            X=np.array([epoch]),
            Y=dice,
            win=self.dice,  # win要保持一致
            update='append')


class loss_visualize(object):
    def __init__(self, title='loss', env='loss'):
        self.viz = Visdom(env=env)
        epoch = 0
        self.title = title
        self.loss = self.viz.line(X=np.array([epoch]),
                                  Y=np.zeros([1, 4]),  # 2 stand train and valid
                                  opts=dict(legend=['train loss', 'valid loss', 'train accuracy', 'valid accuracy'],
                                            showlegend=True, title=self.title))

    def plot_loss(self, epoch, epoch_loss):
        train_loss = epoch_loss['train']
        valid_loss = epoch_loss['valid']
        train_accuracy = epoch_loss['train_accuracy']
        valid_accuracy = epoch_loss['valid_accuracy']
        if valid_loss is not None:
            loss = np.append(train_loss, [valid_loss, train_accuracy, valid_accuracy])[np.newaxis, :]
        else:
            loss = [train_loss]
        self.viz.line(
            X=np.array([epoch]),
            Y=loss,
            win=self.loss,  # win要保持一致
            update='append')
