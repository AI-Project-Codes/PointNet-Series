# -*- encoding: utf-8 -*-
"""
=================================================
@path   : PointNet-Series -> pointnet_utils.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022-01-20 11:02
@Version: v0.1
@License: (C)Copyright 2020-2022, zYx.Tom
@Reference:
@Desc   :
==================================================
"""
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


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


class PointNetEncoder(nn.Module):
    """
    PointNet的编码器
    """

    def __init__(self, global_feat=True, feature_transform=True, channel=3):
        """

        :param global_feat: 分类问题，只生成全局特征；分割问题，拼接生成的局部特征+全局特征
        :param feature_transform: 第二个变换矩阵，特征变换矩阵
        :param channel: 输入数据的通道数
        """
        super(PointNetEncoder, self).__init__()
        self.stn = STNkd(channel=3)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feature = global_feat
        self.feature_transform = feature_transform
        if feature_transform:
            self.fstn = STNkd(channel=64)

    def forward(self, x):
        B, D, N = x.size()
        # 第一个变换矩阵，空间对齐矩阵
        trans_matrix = self.stn(x)
        x = x.transpose(2, 1)

        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
            x = torch.bmm(x, trans_matrix)
            x = torch.cat([x, feature], dim=2)
        else:
            x = torch.bmm(x, trans_matrix)

        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        # 第二个变换矩阵，特征变换矩阵
        if self.feature_transform:
            trans_feature = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feature)
            x = x.transpose(2, 1)
        else:
            trans_feature = None

        point_feature = x
        x = F.relu(self.bn2(self.conv2(x)))

        # 全连接网络+bn
        x = self.bn3(self.conv3(x))
        # 全局池化层
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feature:
            return x, trans_matrix, trans_feature
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, point_feature], 1), trans_matrix, trans_feature


def feature_transform_regularize(trans):
    #
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    I = I.cuda() if trans.is_cuda else I
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


def main(name):
    print(f'Hi, {name}', datetime.now())
    pass


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
