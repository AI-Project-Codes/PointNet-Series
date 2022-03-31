# -*- encoding: utf-8 -*-
"""
=================================================
@path   : PointNet-Series -> PointNetEncoder.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022-02-10 10:03
@Version: v0.1
@License: (C)Copyright 2020-2022, zYx.Tom
@Reference:
@Desc   :
==================================================
"""
from datetime import datetime

import torch
from torch import nn as nn
from torch.nn import functional as F

from model.stnkd import STNkd


class PointNetEncoder(nn.Module):
    """
    PointNet的编码器
    """

    def __init__(self, global_feat=True, feature_transform=True, channel=3):
        """

        :param global_feat: 分类问题，只生成[全局特征]；分割问题，拼接生成的[局部特征+全局特征]
        :param feature_transform: 是否使用第二个变换矩阵（特征变换矩阵）
        :param channel: 输入数据的通道数
        """
        super(PointNetEncoder, self).__init__()
        self.global_feature = global_feat
        self.feature_transform = feature_transform

        self.stn_3d = STNkd(channel=3)
        self.conv1 = torch.nn.Conv1d(channel, 64, (1,))
        self.conv2 = torch.nn.Conv1d(64, 128, (1,))
        self.conv3 = torch.nn.Conv1d(128, 1024, (1,))
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        if feature_transform:
            self.stn_feature = STNkd(channel=64)

    def forward(self, x):
        batch_size, dimension, numbers = x.size()

        # 第一个变换矩阵：空间对齐矩阵，只处理前3维
        trans_matrix, x = self._stn3d(dimension, x)

        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        # 第二个变换矩阵：特征变换矩阵，同时处理多个维度（>3）
        trans_feature, x = self._stnkd(x)

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
            x = x.view(-1, 1024, 1).repeat(1, 1, numbers)
            return torch.cat([x, point_feature], 1), trans_matrix, trans_feature

    def _stnkd(self, x):
        if self.feature_transform:
            trans_feature = self.stn_feature(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feature)
            x = x.transpose(2, 1)
        else:
            trans_feature = None
        return trans_feature, x

    def _stn3d(self, dimension, x):
        """
        用于对齐三维坐标时，输入可能是未经处理的数据，即其特征并未被网络规范，而且其特征可能多于三维，
        需要将多余三维的特征单独提取出来，然后再对三维坐标进行对齐，对齐后再将坐标与其他特征拼接
        :param dimension: 
        :param x: 
        :return: 
        """
        trans_matrix = self.stn_3d(x)
        x = x.transpose(2, 1)
        if dimension > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
            x = torch.bmm(x, trans_matrix)
            x = torch.cat([x, feature], dim=2)
        else:
            x = torch.bmm(x, trans_matrix)
        return trans_matrix, x


def main(name):
    print(f'Hi, {name}', datetime.now())
    pass


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
