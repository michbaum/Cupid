# CUPID Model

## Description

<!-- Share any information you would like others to know. For example:
Author: @xxx.
This is an implementation of \[XXX\]. -->

This project implements an instance pointcloud matching network, i.e., given some prior class & instance labeled pointcloud, the goal of the model is to match individual instances, that belong to the same object in the scene

Take a look at the [configs](./configs/) folder for inspiration how to add new models with your adaptations.

## Usage

### Training commands

In MMDet3D's root directory, run the following command to train the model:

**Normal PointNet++ Matching Model**
```bash
python tools/train.py projects/CUPID/configs/pointnet2_matching.py
```

**Small PointNet++ Matching Model**
```bash
python tools/train.py projects/CUPID/configs/pointnet2_matching_small.py
```

**Normal PointNet++ Semantic Segmentation Model**
```bash
python tools/train.py projects/CUPID/configs/pointnet2_seg_only.py
```

**Normal PointNet++ Panoptic Segmentation Model**
```bash
python tools/train.py projects/CUPID/configs/pointnet2_panoptic.py
```


### Testing commands

In MMDet3D's root directory, run the following command to test the model:

**Normal PointNet++ Matching Model**
```bash
python tools/test.py projects/CUPID/configs/pointnet2_matching.py work_dirs/pointnet2_matching/checkpoint.pth
```

**Small PointNet++ Matching Model**
```bash
python tools/test.py projects/CUPID/configs/pointnet2_matching_small.py work_dirs/pointnet2_matching/checkpoint.pth
```

**Normal PointNet++ Semantic Segmentation Model**
```bash
python tools/test.py projects/CUPID/configs/pointnet2_seg_only.py work_dirs/pointnet2_seg_only/checkpoint.pth
```

**Normal PointNet++ Panoptic Segmentation Model**
```bash
python tools/test.py projects/CUPID/configs/pointnet2_panoptic.py work_dirs/pointnet2_panoptic/checkpoint.pth
```

**Heuristic inference**
```bash
python tools/test.py projects/CUPID/configs/cupid_heuristic_matching.py None
```

## Results


**Heuristic Inference on 1000 scenes dataset**
```
+---------+-----------+-------------+-------------+----------+--------+---------+
| metrics | precision | sensitivity | specificity | f1-score | acc    | acc_cls |
+---------+-----------+-------------+-------------+----------+--------+---------+
| results | 0.5457    | 0.5862      | 0.9832      | 0.5652   | 0.9700 | 0.7847  |
+---------+-----------+-------------+-------------+----------+--------+---------+
```

**Panoptic Segmentation on 1000 scenes dataset (200epochs training)**
```
+---------+-------------+--------+--------+--------+--------+---------+                                
| classes | unannotated | table  | box    | miou   | acc    | acc_cls |                                
+---------+-------------+--------+--------+--------+--------+---------+
| results | nan         | 0.9968 | 0.9942 | 0.9955 | 0.9979 | 0.9976  |
+---------+-------------+--------+--------+--------+--------+---------+
+------------------------+--------+
| Instance Metrics       | Value  |
+------------------------+--------+
| mean_coverage          | 0.8322 |
| mean_weighted_coverage | 0.8892 |
| precision              | 0.7280 |
| recall                 | 0.9233 |
| f1                     | 0.8141 |
+------------------------+--------+
```