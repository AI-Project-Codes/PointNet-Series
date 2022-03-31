"""
=================================================
@path   : PointNet-Series -> pointnet_classes.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022-01-21 14:14
@Version: v0.1
@License: (C)Copyright 2020-2022, zYx.Tom
@Reference:
@Desc   : PointNet 分类器网络
==================================================
"""
from datetime import datetime
import torch.nn as nn
from torch.nn import functional as F
from model.pointnet_encoder import PointNetEncoder


class PointNetCls(nn.Module):
    def __init__(self, num_cls=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feature = PointNetEncoder(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_cls)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        pass

    def forward(self, x):
        x, _, _ = self.feature(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def main(name):
    print(f'Hi, {name}', datetime.now())
    pass


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
