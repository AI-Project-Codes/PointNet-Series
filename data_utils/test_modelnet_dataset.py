"""
=================================================
@path   : PointNet-Series -> test_modelnet_dataset.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022-01-19 15:13
@Version: v0.1
@License: (C)Copyright 2020-2022, zYx.Tom
@Reference:
@Desc   :
==================================================
"""
from argparse import Namespace
from datetime import datetime

from data_utils.modelnet_dataset import ModelNetDataset


class TestClass:
    def test_modelnet10_test_1024pts(self):
        args = Namespace()
        args.uniform_sample = False
        args.modelnet10 = True
        data_path = "../data/pickle/modelnet/"
        myDataset = ModelNetDataset(root=data_path, args=args, split='test')
        assert myDataset.get_pickle_file_name() == "../data/pickle/modelnet/modelnet10_test_1024pts.dat"

    def test_modelnet10_test_1024pts_fps(self):
        args = Namespace()
        args.uniform_sample = True
        args.modelnet10 = True
        data_path = "../data/pickle/modelnet/"
        myDataset = ModelNetDataset(root=data_path, args=args, split='test')
        assert myDataset.get_pickle_file_name() == "../data/pickle/modelnet/modelnet10_test_1024pts_fps.dat"

    def test_modelnet10_train_1024pts(self):
        args = Namespace()
        args.uniform_sample = False
        args.modelnet10 = True
        data_path = "../data/pickle/modelnet/"
        myDataset = ModelNetDataset(root=data_path, args=args, split='train')
        assert myDataset.get_pickle_file_name() == "../data/pickle/modelnet/modelnet10_train_1024pts.dat"

    def test_modelnet10_train_1024pts_fps(self):
        args = Namespace()
        args.uniform_sample = True
        args.modelnet10 = True
        data_path = "../data/pickle/modelnet/"
        myDataset = ModelNetDataset(root=data_path, args=args, split='train')
        assert myDataset.get_pickle_file_name() == "../data/pickle/modelnet/modelnet10_train_1024pts_fps.dat"

    def test_modelnet40_test_1024pts(self):
        args = Namespace()
        args.uniform_sample = False
        args.modelnet10 = False
        data_path = "../data/pickle/modelnet/"
        myDataset = ModelNetDataset(root=data_path, args=args, split='test')
        assert myDataset.get_pickle_file_name() == "../data/pickle/modelnet/modelnet40_test_1024pts.dat"

    def test_modelnet40_test_1024pts_fps(self):
        args = Namespace()
        args.uniform_sample = True
        args.modelnet10 = False
        data_path = "../data/pickle/modelnet/"
        myDataset = ModelNetDataset(root=data_path, args=args, split='test')
        assert myDataset.get_pickle_file_name() == "../data/pickle/modelnet/modelnet40_test_1024pts_fps.dat"

    def test_modelnet40_train_1024pts(self):
        args = Namespace()
        args.uniform_sample = False
        args.modelnet10 = False
        data_path = "../data/pickle/modelnet/"
        myDataset = ModelNetDataset(root=data_path, args=args, split='train')
        assert myDataset.get_pickle_file_name() == "../data/pickle/modelnet/modelnet40_train_1024pts.dat"

    def test_modelnet40_train_1024pts_fps(self):
        args = Namespace()
        args.uniform_sample = True
        args.modelnet10 = False
        data_path = "../data/pickle/modelnet/"
        myDataset = ModelNetDataset(root=data_path, args=args, split='train')
        assert myDataset.get_pickle_file_name() == "../data/pickle/modelnet/modelnet40_train_1024pts_fps.dat"


def main(name):
    print(f'Hi, {name}', datetime.now())

    pass


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
