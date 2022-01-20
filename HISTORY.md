# 项目日志

## 工作日志

- 2022-01-20
  - 创建 pointnet_utils.py 文件
    - 空间变换网络（Spatial Transformer Networks，STN）：STNkd 类
    - PointNet的编码器：PointNetEncoder 类
  - 创建 test_pointnet_utils.py 测试文件
    - test_stn3d()
    - test_stnkd()
    - test_PointNetEncoder_Classification()
    - test_PointNetEncoder_Segmentation()
  - 创建 modelnet_prepare.py 文件
- 2022-01-19
  - 创建仓库：README.md, .gitignore, LICENSE 文件
  - 创建 modelnet_dataset.py 文件
    - ModelNetDataset 类
  - 创建 test_modelnet_dataset.py 测试文件
    - test_modelnet10_test_1024pts()
    - test_modelnet10_test_1024pts_fps()
    - test_modelnet10_train_1024pts()
    - test_modelnet10_train_1024pts_fps()
    - test_modelnet40_test_1024pts()
    - test_modelnet40_test_1024pts_fps()
    - test_modelnet40_train_1024pts()
    - test_modelnet40_train_1024pts_fps()