# Extended KITTI Dataset

This page provides specific tutorials about the usage of MMDetection3D for the extended KITTI dataset.
The dataset can be readily used for object segmentation and matching tasks, and efforts have been made to enable 3D detection and pose estimation tasks in the future, but adaptations are still needed in that area.

## Prepare dataset

You can produce the simulation dataset via the dataset_utils tools and blenderproc scripts found in [this repo](https://github.com/michbaum/bop_dataset_utils/tree/main/learning).

Like the general way to prepare dataset, it is recommended to symlink the dataset root to `$MMDETECTION3D/data`.

The folder structure should be organized as follows before our processing (which will be done for you automatically if utilizing our [export scripts](https://github.com/michbaum/bop_dataset_utils/tree/main/learning/dataset_utils)).

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
Wherein the train/val/trainval/test.txt files contain the scene numbers (subfolders of images/labels/pointclouds) belonging to the respective splits (one per line). Typically, test.txt simply enumerates all the scenes in the testing folder, whilst train & val.txt divide the scenes in the training folders. trainval.txt once again numerates all the scenes in the training folder. There is one calib file per scene.

The **calib** file contains two lines per camera in the scene consisting of the camera intrinsic transformation K (9 elements) and the camera extrinsics (IMU/World to camera, 12 elements, 9 rotation + 3 translation). These are followed by a line containing the IMU/World to lidar extrinsic transformation (12 elements). We are not using a separate IMU/World coordinate system and expect this extrinsic to be identity, but this could be used for extended use-cases (probably needs more adaptations). This setup assumes that all lidar data (irrespective of how many sensors are used in the dataset) is transformed into a unified coordinate system before ingress here.
All parameters are expected to be saved row-wise.

**metadata.txt** contains the number of cameras & pointclouds per scene. If they're the same, we expect the pointclouds to be produced from RGB-D cameras (which influences data preparation down the line). If there is only a single pointcloud, we expect a velodyne LiDAR. Other configurations are not yet supported.
Additionally, it contains the dimension of the pointcloud. By default, we assume the
dimension to be 8 for (x, y, z, r, g, b, class_label_prior, instance_id_prior). Both the class_label and instance_id are assumed to be given by a first stage/2D segmentation approach like Yolov8 and are 'just' initial guesses that we use in our models down the line. During our preparation/in our export scripts, we make sure to make the instance IDs unique/ever increasing over multiple views on the same object within the scene, to not leak ground truth information to the models.
The ground truth class labels and instance/object ids need be provided seperately in the labels folder as shown above (000000.bin in the labels folder) to be used in the loss computations during training. The pointcloud there is expected to consist of 6 channels (class_label_1, class_label_2, class_label_3, instance_id_1, instance_id_2, instance_id_3). For the ground truth case, we expect only one class label and instance id for the moment (every point should naturally have a unique ground truth label), if there are multiple (due to simulation/annotation inaccuracies/rounding), the points will be ignored down the line. This is in accordance with the original KITTI paper (as it indicates ambiguity in the ground truth, which indeed exists with our blenderprocv2 simulation).
Since we save only one class_label & instance_id per point in our pointcloud for inferrance, we simply keep the first labels we receive from different 2D masks as an arbitrary choice (remember, those points are ignored in training & eval anyways).

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

**Note:** This is simply a preparation for future adoptions of the format for 3D detection & pose estimation tasks. These annotations are not really utilized so far & also not completely adapted in the data loading - specifically, work is needed to correctly ingress the 3D bounding boxes with their 3 rotation axes (since mmdetection3d originally only supports 1 yaw axis).

### Create extended KITTI dataset

To create an extended KITTI point cloud dataset, we load the raw point cloud data and generate the relevant annotations including object labels and bounding boxes (not completely adapted). We also generate all single training objects' point clouds in the extended KITTI dataset and save them as `.bin` files in `data/extended_kitti/extended_kitti_gt_database` (again, not completely adapted, depends on correct implementation of the 3D bboxes). Meanwhile, `.pkl` info files are also generated for training and/or validation. Subsequently, create extended KITTI data by running:

```bash
python tools/create_data.py extended_kitti --root-path ./data/extended_kitti --out-dir ./data/extended_kitti --extra-tag extended_kitti
```

Note that if your local disk does not have enough space for saving converted data, you can change the `--out-dir` to anywhere else, and you can add the `--with-plane` flag to utilize ground plane data, if `planes` are prepared (currently not implemented/used).

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
- `kitti_gt_database/xxxxx.bin`: point cloud data included in each 3D bounding box of the training dataset (not fully adapted, also currently not created during data creation since we don't need it for our models).
- `kitti_infos_train.pkl`: training dataset, a dict contains two keys: `metainfo` and `data_list`.
  `metainfo` contains the basic information for the dataset itself, such as `categories`, `dataset` and `info_version`, while `data_list` is a list of dict, each dict (hereinafter referred to as `info`) contains all the detailed information of single sample/scene as follows:
  - info\['scene_idx'\]: The index of this scene in the whole dataset (at some points also saved as sample_idx - interchangeable).
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
    - info\['lidar_points'\]\['num_pts_feats'\]: The feature dimension of the pointclouds. Typically 8: (x, y, z, r, g, b, class_id_prior, instance_id_prior) 
    - info\['lidar_points'\]\['PCx']\['pc_idx'\]: The index of this sample in the scene.
    - info\['lidar_points'\]\['PCx']\['lidar_path'\]: The filename of the lidar point cloud data.
    - info\['lidar_points'\]\['lidar2camx'\]: Transformation from lidar coordinates to camera x coordinates with shape (4, 4).
    - info\['lidar_points'\]\['imu2lidar'\]: Transformation from IMU coordinates to lidar coordinates with shape (4, 4).
  - info\['instances'\]\['CAMx'\]: It is a list of dict. Each dict contains all annotation information of single instance/object within a specific image in the scene. For the i-th instance:
    - info\['instances'\]\['CAMx'\]\[i\]\['bbox'\]: List of 4 numbers representing the 2D bounding box of the instance, in (x1, y1, x2, y2) order.
    - info\['instances'\]\['CAMx'\]\[i\]\['bbox_3d'\]: List of 9 numbers representing the 3D bounding box of the instance, in (x, y, z, l, h, w, rot_x, rot_y, rot_z) order. 
    <!-- TODO: (michbaum) Adapt code that uses the bboxes, making sure that the coordinates are correctly transformed -->
    - info\['instances'\]\['CAMx'\]\[i\]\['bbox_label'\]: An int indicate the 2D label of instance and the -1 indicating ignore.
    - info\['instances'\]\['CAMx'\]\[i\]\['bbox_label_3d'\]: An int indicate the 3D label of instance and the -1 indicating ignore.
    - info\['instances'\]\['CAMx'\]\[i\]\['depth'\]: Projected center depth of the 3D bounding box with respect to the image plane.
    - info\['instances'\]\['CAMx'\]\[i\]\['num_lidar_pts'\]: The number of LiDAR points in the 3D bounding box (not adapted since dependant on correct 3D bboxes).
    - info\['instances'\]\['CAMx'\]\[i\]\['center_2d'\]: Projected 2D center of the 3D bounding box.
    - info\['instances'\]\['CAMx'\]\[i\]\['difficulty'\]: KITTI difficulty: 'Easy', 'Moderate', 'Hard'.
    - info\['instances'\]\['CAMx'\]\[i\]\['truncated'\]: Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries.
    - info\['instances'\]\['CAMx'\]\[i\]\['occluded'\]: Integer (0,1,2,3) indicating occlusion state: 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown.
    - info\['instances'\]\['CAMx'\]\[i\]\['group_ids'\]: Used for multi-part object.
    - info\['instances'\]\['PCx'\]\['pts_mask_path'\]: Path to the ground truth semantic/instance label file. Per default, assume this form: (class_id_1, class_id_2, class_id_3, instance_id_1, instance_id_2, instance_id_3) from a ground truth producer. We only take the first class_id & instance_id as ground truth, but if more are present, the point currently gets ignored due to ambiguity in the gt data.
  - info\['plane'\](optional): Ground level information. (Not used/supported right now)

Please refer to [extended_kitti_converter.py](../../../../tools/dataset_converters/extended_kitti_converter.py) and [update_infos_to_v2.py ](../../../../tools/dataset_converters/update_infos_to_v2.py) for more details.


**Note: Stuff that is not fully adapted yet**
Since we train a panoptic segmentation network (kinda, not really, but similar), we don't necessarily require the bbox, bbox_3d & the labels as well as the num_lidar_pts within the ground truth 3D bboxes & the object scores for our network to train.
Nonetheless, the labels & the difficulty/truncated & occluded scores have been populated already. What we didn't test/adapt fully were the bbox, bbox_3d, depth, num_lidar_pts & center_2d fields. If one would want to utilize
this extended KITTI dataset for object detection/pose estimation tasks, the respective dataloaders would most likely need to be adapted. Consult [create_data.py](../../../../tools/create_data.py), 
[update_infos_to_v2.py](../../../../tools/dataset_converters/update_infos_to_v2.py), [extended_kitti_converte.py](../../../../tools/dataset_converters/extended_kitti_converter.py), 
[extended_kitti_data_utils.py](../../../../tools/dataset_converters/extended_kitti_data_utils.py), [create_gt_database.py](../../../../tools/dataset_converters/create_gt_database.py) for changes regarding the writing of the 
data to disk in preparation and [loading.py](../../../../mmdet3d/datasets/transforms/loading.py) for the loading functions necessary to load the data correctly into a dataset for iteration.


## Train pipeline

A typical train pipeline of 3D segmentation on Extended KITTI Datasets is as below:

```python

# PARAMETERS
num_points = 8192 # Number of points sampled for model input (in each patch sampled in the scene)
num_views_used = 2 # Number of camera views used for the model
pc_dimensions_used = [0, 1, 2, 3, 4, 5, 6, 7] # Point cloud channels used [x, y, z, r, g, b, class_id_prior, instance_id_prior]
# ~ PARAMETERS

train_pipeline = [
    dict(
        type='LoadEKittiPointsFromFile',
        coord_type='LIDAR',
        shift_height=False,
        use_color=True,
        use_prior_labels=True if len(pc_dimensions_used) > 6 else False,
        load_dim=8,
        use_dim=pc_dimensions_used,
        backend_args=backend_args),
    dict(
        type='LoadEKittiAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=False,
        with_panoptic_3d=True, 
        backend_args=backend_args),
    # (michbaum) Sample and combine n pointclouds per scene here producing more samples
    dict(
        type='SampleKViewsFromScene', 
        num_views=num_views_used,
    ),
    # (michbaum) Filter out points that are not in the point_cloud_range -> ROI of the table & boxes
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # (michbaum) Maps class labels newly if needed, depending on the ignore idx etc.
    dict(type='EKittiPointSegClassMapping'),
    # (michbaum) Sampels patches within the whole scene and num_points therein. Also shifts coordinates relative to sample center.
    dict(
        type='IndoorPatchPointSample',
        num_points=num_points,
        block_size=1.5,
        ignore_index=len(class_names),
        use_normalized_coord=False,
        enlarge_size=0.2,
        min_unique_num=None),
    # (michbaum) Normalizes color to [0, 1] -> Does NOT compute mean color in pointcloud or something
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
    dict(type='GlobalRotScaleTrans',
         rot_range=[-1.5708, 1.5708],
         scale_ratio_range=[0.95, 1.05]),
    # dict(type='PointShuffle'), DO NOT USE
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
]
```

- Data augmentation:
  - `RandomFlip3D`: randomly flip input point cloud horizontally or vertically.
  - `GlobalRotScaleTrans`: rotate and scale input point cloud.
  - `PointShuffle`: Careful: This augmentation seems to deteriote performance greatly!

For 3D object matching, the train pipeline looks like this:
```python
TODO: Insert example here
```

## Evaluation

An example to evaluate PointNet++ with the segmentation metric is as follows:

```shell
python /path/to/mmdetection3d/tools/test.py /path/to/CUPID/configs/pointnet2_seg_only.py /path/to/checkpoint.pth
```

## Metrics

For segmentation models, we utilize the [ScanNet segmentation metric](https://kaldir.vc.in.tum.de/scannet_benchmark/documentation) of MMDetection3D:
```
+---------+------------+--------+--------+--------+--------+---------+
| classes | background | table  | box    | miou   | acc    | acc_cls |
+---------+------------+--------+--------+--------+--------+---------+
| results | 0.0000     | 0.9858 | 0.9823 | 0.6560 | 0.9865 | 0.6651  |
+---------+------------+--------+--------+--------+--------+---------+
```

For object matching, we utilize TODO: (michbaum)
