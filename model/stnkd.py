# -*- encoding: utf-8 -*-
"""
=================================================
@path   : PointNet-Series -> stnkd.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022-02-10 10:01
@Version: v0.1
@License: (C)Copyright 2020-2022, zYx.Tom
@Reference: https://arxiv.org/abs/1506.02025
@Desc   : 空间变换网络（Spatial Transformer Networks，STN）
==================================================
"""
from datetime import datetime

import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class STNkd(nn.Module):
    """
    空间变换网络（Spatial Transformer Networks，STN）
    """

    def __init__(self, channel=3, k=3):
        """
        :param channel: 输入数据的通道数，特征变换的维度
        """
        super(STNkd, self).__init__()

        self.channel = channel

        self.conv1 = torch.nn.Conv1d(channel, 64, (1,))
        self.conv2 = torch.nn.Conv1d(64, 128, (1,))
        self.conv3 = torch.nn.Conv1d(128, 1024, (1,))

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, channel * channel)

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        pass

    def forward(self, x):
        """
        :return: 空间变换矩阵
        """
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # 单位矩阵 展开成 序列
        identify_list = Variable(torch.eye(self.channel).view(1, -1).repeat(batch_size, 1))
        identify_list = identify_list.cuda() if x.is_cuda else identify_list
        x = x + identify_list
        x = x.view(-1, self.channel, self.channel)
        return x


def main(name):
    print(f'Hi, {name}', datetime.now())
    pass


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
