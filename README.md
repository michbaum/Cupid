# Rapid, multi-view synthesis for 3D object segmentation and pose estimation for manipulation tasks

This repository is based of and extends the excellent [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) repository and introduces a novel 3D object segmentation and pose estimation network for Indoor tasks based on the sensor data of multiple depth cameras.

## Setup

The original mmdetection3d readme with installation instructions can be found [here](ORIG_README.md).

Debug notes:
- if facing libGl errors after setup when trying out the first example, run:
    ```
    conda install -c conda-forge libstdcxx-ng
    ```
    in your conda environment.

