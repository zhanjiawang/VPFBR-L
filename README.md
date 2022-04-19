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
#voxel plane features based registration
./VPFBR ../data/1410.pcd ../data/3540.pcd
#voxel plane features based localization
./VPFBL ../data/1410.pcd ../data/3540.pcd
```
