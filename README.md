# VPFBR-L
A 3D point cloud registration and localization method based on voxel plane features

## Usage
#### 1. Requirement
```
1. cmake
2. PCL
3. Ceres
```

#### 2. Build
```
cd ${YOUR_PROJECT_DIR}
mdkir build
cd build
cmake ..
make
```

#### 3. Run
```
#voxel plane features based registration
./VPFBR ../data/1410.pcd ../data/3540.pcd
#voxel plane features based localization
./VPFBL ../data/1410.pcd ../data/3540.pcd
```

#### 3. Paper
```
#If you used this open source code, you can refer to our paper
【1】李建微, 占家旺. 三维点云配准方法研究进展[J]. 中国图象图形学报, 2022, 27(02): 349-367. 
【2】Jianwei Li, Jiawang Zhan, Ting Zhou, Virgílio A. Bento, Qianfeng Wang. Point Cloud Registration and Localization Based on Voxel Plane Features [J]. ISPRS Journal of Photogrammetry and Remote Sensing.（expected to be published in this journal）
```
