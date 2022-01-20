"""
=================================================
@path   : PointNet-Series -> modelnet_prepare.py
@IDE    : PyCharm
@Author : zYx.Tom, 526614962@qq.com
@Date   : 2022-01-19 15:54
@Version: v0.1
@License: (C)Copyright 2020-2022, zYx.Tom
@Reference:
https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
@Desc   : 预处理数据，加工多个以文件保存的数据为单个列表保存的数据
==================================================
"""
import pickle
import numpy as np
import argparse
from datetime import datetime
from os.path import join

from tqdm import tqdm


def farthest_point_sample(xyz, npoint):
    """
    最远点采样(N:Number, D:Dimension)
    不断迭代地选择距离已有采样点集合的最远点，从而得到物体的轮廓
    采样点的数量应该小于原始点的数量，否则就是取出所有的点
    Input:
        xyz: point cloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled point cloud index, [npoint, D]
    """
    N, D = xyz.shape
    xyz = xyz[:, :3]
    centroids = np.zeros((npoint,))  # 质心点
    distance = np.ones((N,)) * 1e10  # 采样点到所有点的距离最大值
    # farthest = np.random.randint(0, N)  # 随机初始化一个点作为首个质心点

    barycenter = (np.sum(xyz, 0) / xyz.shape[0])  # 计算形状的质心坐标 及 距离质心最远的点
    dist = np.sum((xyz - barycenter) ** 2, -1)  # 计算所有点到形状质心的距离
    farthest = np.argmax(dist, -1)  # 将距离质心最远的点作为第一个点（比随机初始化更加精确）

    # 每一次循环找的是当前质心点的最远点
    # distance里存的是每一次操作以后N个点相对于所有备选点集的最远距离
    # mask存在是因为点到点集的距离是点到点集中每个点的距离中的最小值
    for i in range(npoint):
        centroids[i] = farthest  # 更新第 i 个最远点
        centroid = xyz[farthest, :]  # 取出这个最远点的xyz坐标
        dist = np.sum((xyz - centroid) ** 2, -1)  # 计算点集中所有的点到这个质心点的距离
        mask = dist < distance
        distance[mask] = dist[mask]  # 更新最远点与整个点集之间的最短距离
        farthest = np.argmax(distance, -1)  # 返回最远点的索引
    xyz = xyz[centroids.astype(np.int32)]  # 返回坐标
    return xyz


def main(name):
    print(f'Hi, {name}', datetime.now())
    prepare_data()
    # origin_xyz = np.random.randn(5, 3)
    # print(origin_xyz)
    # print(farthest_point_sample(origin_xyz, 3))


def prepare_data():
    args = argparse.Namespace()
    print("将 ModelNet 数据加工成一个数组，方便存储和使用")
    root = "../data/modelnet40_normal_resampled"
    num_points = 1024
    modelnet_cat = "modelnet10"
    process_data(root, num_points, modelnet_cat, split='train', fps='')
    process_data(root, num_points, modelnet_cat, split='test', fps='')
    process_data(root, num_points, modelnet_cat, split='train', fps='_fps')
    process_data(root, num_points, modelnet_cat, split='test', fps='_fps')
    modelnet_cat = "modelnet40"
    process_data(root, num_points, modelnet_cat, split='train', fps='')
    process_data(root, num_points, modelnet_cat, split='test', fps='')
    process_data(root, num_points, modelnet_cat, split='train', fps='_fps')
    process_data(root, num_points, modelnet_cat, split='test', fps='_fps')


def process_data(root, num_points, modelnet_cat, split, fps):
    """
    将modelnet中的数据加工到列表变量中，方便存储与使用
    :param root: 数据所在目录
    :param num_points:
    :param modelnet_cat: 数据是10个类别，还是40个类别
    :param split: 数据用于训练（train）还是测试（test）
    :param fps: 使用最远点采样（farthest point sample）
    :return: 处理成功或者失败
    """
    # 从 modelnet10_shape_names.txt，modelnet40_shape_names.txt 取出形状类别
    shape_category = [line.rstrip() for line in open(join(root, f"{modelnet_cat}_shape_names.txt"))]
    # 对形状类别进行编号
    shape_classes = dict(zip(shape_category, range(len(shape_category))))
    # 从 modelnet10_train.txt, modelnet10_test.txt, modelnet40_train.txt, modelnet40_test.txt 中取出形状列表
    shape_ids = [line.rstrip() for line in open(join(root, f"{modelnet_cat}_{split}.txt"))]
    # 将列表中每个形状的类别提出
    shape_names = ['_'.join(file_name.split('_')[0:-1]) for file_name in shape_ids]
    # 生成每个形状文件的访问路径
    data_path = [(shape_names[i], join(root, shape_names[i], shape_ids[i] + '.txt'))
                 for i in range(len(shape_ids))]
    # 保存 pickle 文件的路径
    save_path_str = f"../data/tmp/{modelnet_cat}_{split}_{num_points}pts{fps}.dat"
    # 初始化保存所有数据（点、标签）的列表变量
    num_data = len(shape_ids)
    list_of_points = [None] * num_data
    list_of_labels = [None] * num_data
    # 读入所有数据
    for index in tqdm(range(num_data), total=num_data):
        file_name = data_path[index]
        shape_class = np.array([shape_classes[data_path[index][0]]]).astype(np.int32)
        point_set = np.loadtxt(file_name[1], delimiter=',').astype(np.float32)
        # 按默认顺序采样，还是按最远点采样
        list_of_points[index] = point_set[0:num_points, :] \
            if fps == "" else farthest_point_sample(point_set, num_points)
        list_of_labels = shape_class
        pass
    # 写入文件中
    with open(save_path_str, 'wb') as f:
        pickle.dump([list_of_points, list_of_labels], f)


if __name__ == "__main__":
    __author__ = 'zYx.Tom'
    main(__author__)
