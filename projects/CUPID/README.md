# CUPID Model

## Description

<!-- Share any information you would like others to know. For example:
Author: @xxx.
This is an implementation of \[XXX\]. -->

This project implements an instance pointcloud matching network, i.e., given some prior class & instance labeled pointcloud, the goal of the model is to match individual instances, that belong to the same object in the scene

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