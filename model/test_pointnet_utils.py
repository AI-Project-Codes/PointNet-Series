"""
=================================================
@path   : PointNet-Series -> test_pointnet_utils.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022-01-19 15:13
@Version: v0.1
@License: (C)Copyright 2020-2022, zYx.Tom
@Reference:
@Desc   :
==================================================
"""
import unittest

import torch
from torch.autograd import Variable

from model.pointnet_utils import feature_transform_regularize
from model.pointnet_encoder import PointNetEncoder
from model.stnkd import STNkd
from tools import log_title, log_subtitle, log_info, log_debug
import logging

logging.basicConfig(
        level=logging.DEBUG,
        filename='../log/model/pointnet_utils.log',
        filemode='w',
        format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
)


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 32
        self.num_points = 2500

    def log_method(self, title, sim_data, out):
        log_subtitle(title)
        log_debug(f"sim_data.size={sim_data.size()}")
        log_debug(f"stn.size={out.size()}")
        try:
            log_debug(f"loss={feature_transform_regularize(out)}")
        except Exception:
            log_debug(f"loss=null")
            pass

    def test_stn3d(self):
        # 批处理的大小：32条数据；3个特征；每条数据的数据点个数：2500个点
        title = "stn3d-data"
        channel = 3
        sim_data = Variable(torch.rand(self.batch_size, channel, self.num_points))
        trans = STNkd(channel)
        out = trans(sim_data)
        self.log_method(title, sim_data, out)
        self.assertEqual(out.detach().numpy().shape, (self.batch_size, channel, channel), "特征变换矩阵")

    def test_stnkd(self):
        # 批处理的大小：32条数据；64个特征；每条数据的数据点个数：2500个点
        title = "stnkd-data"
        channel = 64
        sim_data = Variable(torch.rand(self.batch_size, channel, self.num_points))
        trans = STNkd(channel)
        out = trans(sim_data)
        self.log_method(title, sim_data, out)
        self.assertEqual(out.detach().numpy().shape, (self.batch_size, channel, channel), "特征变换矩阵")

    def test_PointNetEncoder_Classification(self):
        # 批处理的大小：32条数据；3个特征；每条数据的数据点个数：2500个点
        title = "PointNetEncoder"
        in_channel, out_channel, trans_channel = 3, 1024, 64
        sim_data = Variable(torch.rand(self.batch_size, in_channel, self.num_points))

        net_encoder = PointNetEncoder(global_feat=True, feature_transform=True, channel=in_channel)
        out_encoder, trans_matrix, trans_feature = net_encoder(sim_data)
        self.log_method(title, sim_data, out_encoder)
        self.assertEqual(out_encoder.detach().numpy().shape,
                         (self.batch_size, out_channel),
                         "分类问题，增加第二个特征变换矩阵，全局特征")

        net_encoder = PointNetEncoder(global_feat=True, feature_transform=False, channel=in_channel)
        out_encoder, trans_matrix, trans_feature = net_encoder(sim_data)
        self.log_method(title, sim_data, out_encoder)
        self.assertEqual(out_encoder.detach().numpy().shape,
                         (self.batch_size, out_channel),
                         "分类问题，没有第二个特征变换矩阵，全局特征")

    def test_PointNetEncoder_Segmentation(self):
        # 批处理的大小：32条数据；3个特征；每条数据的数据点个数：2500个点
        title = "PointNetEncoder"
        in_channel, out_channel, trans_channel = 3, 1024, 64
        sim_data = Variable(torch.rand(self.batch_size, in_channel, self.num_points))

        net_encoder = PointNetEncoder(global_feat=False, feature_transform=True, channel=in_channel)
        out_encoder, trans_matrix, trans_feature = net_encoder(sim_data)
        self.log_method(title, sim_data, out_encoder)
        self.assertEqual(out_encoder.detach().numpy().shape,
                         (self.batch_size, out_channel + trans_channel, self.num_points),
                         "分割问题，增加第二个特征变换矩阵，全局特征+局部特征")

        net_encoder = PointNetEncoder(global_feat=False, feature_transform=False, channel=in_channel)
        out_encoder, trans_matrix, trans_feature = net_encoder(sim_data)
        self.log_method(title, sim_data, out_encoder)
        self.assertEqual(out_encoder.detach().numpy().shape,
                         (self.batch_size, out_channel + trans_channel, self.num_points),
                         "分割问题，没有第二个特征变换矩阵，全局特征")

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    log_title("stnkd-model")
    global_feature = True
    unittest.main()
