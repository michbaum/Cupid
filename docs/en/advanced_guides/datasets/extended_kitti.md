# Extended KITTI Dataset

This page provides specific tutorials about the usage of MMDetection3D for the extended KITTI dataset.

## Prepare dataset

You can produce the dataset via the dataset_utils tools and blenderproc scripts found in [this repo](https://github.com/michbaum/robin2).

Like the general way to prepare dataset, it is recommended to symlink the dataset root to `$MMDETECTION3D/data`.

The folder structure should be organized as follows before our processing.

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── extended_kitti
│   │   ├── ImageSets
│   │   │   ├── metadata.txt
│   │   │   ├── test.txt
│   │   │   ├── train.txt
│   │   │   ├── trainval.txt
│   │   │   ├── val.txt
│   │   ├── testing
│   │   │   ├── calib
│   │   │   │   ├── 000000.txt
│   │   │   │   ├── 000001.txt
│   │   │   │   ├── ...
│   │   │   ├── images
│   │   │   │   ├── 000000
│   │   │   │   │   ├── 000000.jpg
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   │   ├── pointclouds
│   │   │   │   ├── 000000
│   │   │   │   │   ├── 000000.bin
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   ├── training
│   │   │   ├── calib
│   │   │   │   ├── 000000.txt
│   │   │   │   ├── 000001.txt
│   │   │   │   ├── ...
│   │   │   ├── images
│   │   │   │   ├── 000000
│   │   │   │   │   ├── 000000.jpg
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   │   ├── labels
│   │   │   │   ├── 000000
│   │   │   │   │   ├── 000000.txt
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── 000000.bin
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   │   ├── pointclouds
│   │   │   │   ├── 000000
│   │   │   │   │   ├── 000000.bin
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
```
<!-- TODO: (michbaum) Adapt folder structures if it changes -->
Wherein the train/val/trainval/test.txt files contain the scene numbers (subfolders of images/labels/pointclouds) belonging to the respective splits (one per line). There is one calib file per scene.

The **calib** file contains two lines per camera in the scene consisting of the camera intrinsic transformation K (9 elements) and the camera extrinsics (IMU/World to camera, 12 elements, 9 rotation + 3 translation). These are followed by a line containing the IMU/World to lidar extrinsic transformation (12 elements). We are not using a separate IMU/World coordinate system and expect this extrinsic to be identity, but this could be used for extended use-cases.
All parameters are expected to be saved row-wise.

**metadata.txt** contains the number of cameras & pointclouds per scene. If they're the same, we expect the pointclouds to be produced from RGB-D cameras (which influences data preparation down the line). If there is only a single pointcloud, we expect a velodyne LiDAR. Other configurations are not yet supported.
Additionally, it contains the dimension of the pointcloud. By default, we assume the
dimension to be 8 for (x, y, z, r, g, b, class_label, instance_id). Both the class_label and instance_id are assumed to be given by a first stage/2D segmentation approach like Yolov8 and are 'just' initial guesses.
The ground truth class labels and instance/object ids need be provided seperately in the labels folder as shown above (000000.bin in the labels folder) to be used in the loss computations during training. The pointcloud there is expected to consist of 6 channels (class_label_1, class_label_2, class_label_3, instance_id_1, instance_id_2, instance_id_3). For the ground truth case, we expect only one class label and instance id for the moment, if there are multiple, the points will be ignored down the line. This is in accordance with the original KITTI paper (as it indicates ambiguity in the ground truth, which indeed exists with our blenderprocv2 simulation).

**metadata format**:
```
num_cameras_per_scene: x [int]
num_pointclouds_per_scene: y [int]
pointcloud_dimension: z [int]
depth_scale: d [float]
```

The **label** text files contain the ground truth information about all the objects in the respective images/pointclouds in the following format:

```
name truncated occluded alpha 2d_bbox dimensions location rotation_z rotation_y rotation_x
```

Where:
- name: Object name. Str.
- truncated: Truncation of the object. Float in [0, 1]. 0 is not truncated/fully in frame.
- occluded: Occlusion of the object. Int in [0, 3]. 0 is fully visible, 3 is fully occluded. Note that this is a fraction of the part actually in the frame (so an object can be truncated but still fully visible/not occluded by another object).
- alpha: Observation angle of the object. Float in [-pi, pi].
- 2d_bbox: Bounding box of the object in the camera frame. 4 floats in pixel coordinates (x_1, y_1, x_2, y_2).
- dimensions: Dimensions of the object. 3 floats in [m] for height, width & length.
- location: Location of the object in the LiDAR/world frame. 3 floats in [m] for x, y & z.
- rotation_z: Rotation around the z-axis, aka yaw. Float in [-pi, pi].
- rotation_y: Rotation around the y-axis, aka pitch. Float in [-pi, pi].
- rotation_x: Rotation around the x-axis, aka roll. Float in [-pi, pi].

and the rotation is expected to be in zyx-Euler Angles and everything in LiDAR/PointCloud frame.

```
LiDAR/PointCloud frame:

                    (pitch=0.5*pi, roll=0) up z    x front (pitch=0, yaw=0)
                                              ^   ^
                                              |  /
                                              | /
    (roll=-0.5*pi, yaw=0.5*pi) left y <------ 0
```

<!-- (michbaum) Truncation/occlusion need be populated when exporting, mmdet3d scripts only check the values for classification into different difficulty classes. Maybe ignore at the start. -->

Objects in the scene that should be ignored can be marked as follows:
```
DontCare -1 -1 -10 559.62 175.83 575.40 183.15 -1 -1 -1 -1000 -1000 -1000 -10 -10 -10
```
only providing the bbox. They are expected to come last in the labels file, after all objects that should be considered.

Dimension is expected to be in hwl format.

### Create extended KITTI dataset

To create an extended KITTI point cloud dataset, we load the raw point cloud data and generate the relevant annotations including object labels and bounding boxes. We also generate all single training objects' point clouds in the extended KITTI dataset and save them as `.bin` files in `data/extended_kitti/extended_kitti_gt_database`. Meanwhile, `.pkl` info files are also generated for training or validation. Subsequently, create extended KITTI data by running:

```bash
python tools/create_data.py extended_kitti --root-path ./data/extended_kitti --out-dir ./data/extended_kitti --extra-tag extended_kitti --with-plane
```

Note that if your local disk does not have enough space for saving converted data, you can change the `--out-dir` to anywhere else, and you need to remove the `--with-plane` flag if `planes` are not prepared (currently not implemented).

The folder structure after processing should be as below

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── extended_kitti
│   │   ├── ImageSets
│   │   │   ├── test.txt
│   │   │   ├── train.txt
│   │   │   ├── trainval.txt
│   │   │   ├── val.txt
│   │   ├── testing
│   │   │   ├── calib
│   │   │   │   ├── 000000.txt
│   │   │   │   ├── 000001.txt
│   │   │   │   ├── ...
│   │   │   ├── images
│   │   │   │   ├── 000000
│   │   │   │   │   ├── 000000.jpg
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   │   ├── pointclouds
│   │   │   │   ├── 000000
│   │   │   │   │   ├── 000000.bin
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   ├── training
│   │   │   ├── calib
│   │   │   │   ├── 000000.txt
│   │   │   │   ├── 000001.txt
│   │   │   │   ├── ...
│   │   │   ├── images
│   │   │   │   ├── 000000
│   │   │   │   │   ├── 000000.jpg
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   │   ├── labels
│   │   │   │   ├── 000000
│   │   │   │   │   ├── 000000.txt
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── 000000.bin
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   │   ├── pointclouds
│   │   │   │   ├── 000000
│   │   │   │   │   ├── 000000.bin
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   ├── extended_kitti_gt_database
│   │   │   ├── xxxxx.bin
│   │   ├── extended_kitti_infos_train.pkl
│   │   ├── extended_kitti_infos_val.pkl
│   │   ├── extended_kitti_dbinfos_train.pkl
│   │   ├── extended_kitti_infos_test.pkl
│   │   ├── extended_kitti_infos_trainval.pkl
```

<!-- TODO: (michbaum) Change the following information according to the changes -->
- `kitti_gt_database/xxxxx.bin`: point cloud data included in each 3D bounding box of the training dataset.
- `kitti_infos_train.pkl`: training dataset, a dict contains two keys: `metainfo` and `data_list`.
  `metainfo` contains the basic information for the dataset itself, such as `categories`, `dataset` and `info_version`, while `data_list` is a list of dict, each dict (hereinafter referred to as `info`) contains all the detailed information of single sample as follows:
  - info\['scene_idx'\]: The index of this scene in the whole dataset.
  - info\['images'\]: Information of images captured by multiple cameras in the scene. A dict contains a variable number of keys named `CAMx` for every camera in the scene.
    - info\['images'\]\['CAMx'\]: Include some information about the `CAMx` camera sensor.
      - info\['images'\]['CAMx'\]\['image_idx'\]: The index of this sample in the scene.
      - info\['images'\]\['CAMx'\]\['img_path'\]: The filename of the image.
      - info\['images'\]\['CAMx'\]\['height'\]: The height of the image.
      - info\['images'\]\['CAMx'\]\['width'\]: The width of the image.
      - info\['images'\]\['CAMx'\]\['cam2img'\]: Intrinsic matrix of the camera with shape (3, 4) (extend with a column of 0's).
      - info\['images'\]\['CAMx'\]\['lidar2cam'\]: Transformation matrix from lidar coordinates to camera coordinates with shape (4, 4).
      - info\['images'\]\['CAMx'\]\['lidar2img'\]: Transformation matrix from lidar points to image points with shape (4, 4).
  - info\['lidar_points'\]: A dict containing all the information related to the lidar points.
    - info\['lidar_points'\]\['num_pts_feats'\]: The feature dimension of the pointclouds. Typically 7: (x, y, z, r, g, b, label) 
    - info\['lidar_points'\]\['PCx']\['pc_idx'\]: The index of this sample in the scene.
    - info\['lidar_points'\]\['PCx']\['lidar_path'\]: The filename of the lidar point cloud data.
    - info\['lidar_points'\]\['lidar2camx'\]: Transformation from lidar coordinates to camera x coordinates with shape (4, 4).
    - info\['lidar_points'\]\['imu2lidar'\]: Transformation from IMU coordinates to lidar coordinates with shape (4, 4).
  - info\['instances'\]\['CAMx'\]: It is a list of dict. Each dict contains all annotation information of single instance within a specific image in the scene. For the i-th instance:
    - info\['instances'\]\['CAMx'\]\[i\]\['bbox'\]: List of 4 numbers representing the 2D bounding box of the instance, in (x1, y1, x2, y2) order.
    - info\['instances'\]\['CAMx'\]\[i\]\['bbox_3d'\]: List of 9 numbers representing the 3D bounding box of the instance, in (x, y, z, l, h, w, rot_x, rot_y, rot_z) order. 
    <!-- TODO: (michbaum) Adapt code that uses the bboxes, making sure that the coordinates are correctly transformed -->
    - info\['instances'\]\['CAMx'\]\[i\]\['bbox_label'\]: An int indicate the 2D label of instance and the -1 indicating ignore.
    - info\['instances'\]\['CAMx'\]\[i\]\['bbox_label_3d'\]: An int indicate the 3D label of instance and the -1 indicating ignore.
    - info\['instances'\]\['CAMx'\]\[i\]\['depth'\]: Projected center depth of the 3D bounding box with respect to the image plane.
    - info\['instances'\]\['CAMx'\]\[i\]\['num_lidar_pts'\]: The number of LiDAR points in the 3D bounding box.
    - info\['instances'\]\['CAMx'\]\[i\]\['center_2d'\]: Projected 2D center of the 3D bounding box.
    - info\['instances'\]\['CAMx'\]\[i\]\['difficulty'\]: KITTI difficulty: 'Easy', 'Moderate', 'Hard'.
    - info\['instances'\]\['CAMx'\]\[i\]\['truncated'\]: Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries.
    - info\['instances'\]\['CAMx'\]\[i\]\['occluded'\]: Integer (0,1,2,3) indicating occlusion state: 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown.
    - info\['instances'\]\['CAMx'\]\[i\]\['group_ids'\]: Used for multi-part object.
  - info\['plane'\](optional): Ground level information.

Please refer to [extended_kitti_converter.py](../../../../tools/dataset_converters/extended_kitti_converter.py) and [update_infos_to_v2.py ](../../../../tools/dataset_converters/update_infos_to_v2.py) for more details.


## Train pipeline
<!-- TODO: (michbaum) Change accordingly -->

THIS IS NOT YET ADAPTED

A typical train pipeline of 3D detection on KITTI is as below:

```python
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4, # x, y, z, intensity
        use_dim=4),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
```

- Data augmentation:
  - `ObjectNoise`: apply noise to each GT objects in the scene.
  - `RandomFlip3D`: randomly flip input point cloud horizontally or vertically.
  - `GlobalRotScaleTrans`: rotate input point cloud.

## Evaluation
<!-- TODO: (michbaum) Change accordingly -->

An example to evaluate PointPillars with 8 GPUs with kitti metrics is as follows:

```shell
bash tools/dist_test.sh configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py work_dirs/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class/latest.pth 8
```

## Metrics
<!-- TODO: (michbaum) Change accordingly -->

KITTI evaluates 3D object detection performance using mean Average Precision (mAP) and Average Orientation Similarity (AOS), Please refer to its [official website](http://www.cvlibs.net/datasets/kitti/eval_3dobject.php) and [original paper](http://www.cvlibs.net/publications/Geiger2012CVPR.pdf) for more details.

We also adopt this approach for evaluation on KITTI. An example of printed evaluation results is as follows:

```
Car AP@0.70, 0.70, 0.70:
bbox AP:97.9252, 89.6183, 88.1564
bev  AP:90.4196, 87.9491, 85.1700
3d   AP:88.3891, 77.1624, 74.4654
aos  AP:97.70, 89.11, 87.38
Car AP@0.70, 0.50, 0.50:
bbox AP:97.9252, 89.6183, 88.1564
bev  AP:98.3509, 90.2042, 89.6102
3d   AP:98.2800, 90.1480, 89.4736
aos  AP:97.70, 89.11, 87.38
```
