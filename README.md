# Cupid: Find your perfect 3D match

This repository contains my Master Thesis project concerned with new 3D object matching approaches based on a multi-RGB-D camera indoor setup with large view overlap.

## Rapid, multi-view synthesis for 3D object segmentation, matching and pose estimation for manipulation tasks

The repository is based of and extends the excellent [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) repository and introduces a novel 3D object segmentation, matching and pose estimation network for Indoor tasks based on the sensor data of multiple depth cameras specifically for robotic manipulation tasks.

## Setup

The original mmdetection3d readme with installation instructions can be found [here](ORIG_README.md).

Debug notes:
- if facing libGl errors after setup when trying out the first example, run:
    ```
    conda install -c conda-forge libstdcxx-ng
    ```
    in your conda environment.

