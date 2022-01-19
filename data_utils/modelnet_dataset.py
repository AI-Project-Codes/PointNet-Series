"""
=================================================
@path   : PointNet-Series -> modelnet_dataset.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022-01-19 10:26
@Version: v0.1
@License: (C)Copyright 2020-2022, zYx.Tom
@Reference:
@Desc   :
==================================================
"""
import argparse
import logging
from datetime import datetime
import torch.utils.data as data
from os.path import join
import pickle
from torch.utils.data import DataLoader


class ModelNetDataset(data.Dataset):
    def __init__(self, root, args, split='train'):
        prefix = "modelnet10" if args.modelnet10 else "modelnet40"
        fps = "_fps" if args.uniform_sample else ""
        self.pickle_file_name = join(root, f'{prefix}_{split}_1024pts{fps}.dat')
        try:
            with open(self.pickle_file_name, 'rb') as f:
                self.list_of_points, self.list_of_labels = pickle.load(f)
        except FileNotFoundError:
            print("请先完成数据预处理！")

    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, item):
        return self.list_of_points[item], self.list_of_labels[item]

    def get_pickle_file_name(self):
        return self.pickle_file_name


def main(name):
    """测试ModelNetDataset类"""
    print(f'Hi, {name}', datetime.now())

    """参数"""
    args = argparse.Namespace()
    args.uniform_sample = False
    args.modelnet10 = True

    def log_string(info):
        logger.info(info)
        print(info)
        pass

    """日志"""
    logger = logging.getLogger("ModelNet")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s -%(levelname)s - %(message)s")
    file_handler = logging.FileHandler("../log/data/ModelNet/eval.txt")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string("参数。。。")
    log_string(args)

    """数据加载"""
    log_string("载入数据集。。。")
    data_path = "../data/pickle/modelnet/"
    myDataset = ModelNetDataset(root=data_path, args=args, split='test')
    print(myDataset.get_pickle_file_name())
    data_iter = DataLoader(myDataset, batch_size=15, shuffle=True)
    for i_batch, batch_data in enumerate(data_iter):
        print(i_batch)
        print(batch_data[0])
        print(batch_data[1])
        if i_batch > 2:
            break
    pass


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
