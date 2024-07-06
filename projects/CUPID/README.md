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

**normal pointnet++ panoptic segmentation model**
```bash
python tools/train.py projects/cupid/configs/pointnet2_panoptic_seg.py
```

**3DCupid panoptic matching model**
```bash
python tools/train.py projects/cupid/configs/pointnet2_panoptic_matching.py
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
python tools/test.py projects/CUPID/configs/pointnet2_panoptic_seg.py work_dirs/pointnet2_panoptic_seg/checkpoint.pth
```

**Heuristic inference**
```bash
python tools/test.py projects/CUPID/configs/cupid_heuristic_matching.py None
```

**3DCupid panoptic matching model**
```bash
python tools/test.py projects/cupid/configs/pointnet2_panoptic_matching.py work_dirs/pointnet2_panoptic_matching/checkpoint.pth
```

## Results


**Heuristic Inference on 1000 scenes dataset**
```
+---------+-----------+-------------+-------------+----------+--------+---------+
| metrics | precision | sensitivity | specificity | f1-score | acc    | acc_cls |
+---------+-----------+-------------+-------------+----------+--------+---------+
| results | 0.6052    | 0.5861      | 0.9869      | 0.5955   | 0.9700 | 0.7847  |
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

**3DCupid Panoptic Matching on 1000 scenes dataset (200epochs training)**

+---------+-----------+-------------+-------------+----------+--------+---------+
| metrics | precision | sensitivity | specificity | f1-score | acc    | acc_cls |
+---------+-----------+-------------+-------------+----------+--------+---------+
| results | 0.9165    | 0.9744      | 0.9970      | 0.9446   | 0.9960 | 0.9865  |
+---------+-----------+-------------+-------------+----------+--------+---------+