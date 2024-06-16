# Dataset adapting mmdet3d/datasets/kitti_dataset.py for the extended KITTI dataset.
from typing import Callable, List, Union
from os import path as osp

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.structures import CameraInstance3DBoxes
from .det3d_dataset import Det3DDataset

# TODO: (michbaum) Adapt this class to the extended KITTI dataset.
# Namely need more rotation information and other classes.

@DATASETS.register_module()
class ExtendedKittiDataset(Det3DDataset):
    r"""Extended KITTI Dataset.

    This class serves as the API for experiments on the `KITTI Dataset
    <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>`_.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_lidar=True).
        default_cam_key (str): The default camera name adopted.
            Defaults to 'CAM2'.
        load_type (str): Type of loading mode. Defaults to 'frame_based'.

            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
              to convert to the FOV-based data type to support image-based
              detector.
            - 'fov_image_based': Only load the instances inside the default
              cam, and need to convert to the FOV-based data type to support
              image-based detector.
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes.
            Defaults to [0, -40, -3, 70.4, 40, 0.0].
    """
    # TODO: (michbaum) Change to just using boxes?
    #       Currently label 0 is unnannotated/background, 1 is table and 2 is boxes
    METAINFO = {
        'classes': ('background', 'table', 'box'),
        'palette': [(106, 0, 228), (255, 77, 255), (255, 0, 0)]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True),
                 default_cam_key: str = 'CAM0', # TODO: (michbaum) Check what this is used for exactly
                 load_type: str = 'frame_based', # TODO: (michbaum) Investiage, scene_based if possible
                 box_type_3d: str = 'LiDAR', # TODO: (michbaum) Need to introduce a nove 3D box type
                 filter_empty_gt: bool = True, # TODO: (michbaum) Since our points in gt is bogus, check this
                 test_mode: bool = False,
                 pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0], # TODO: (michbaum) Definitely too big
                 **kwargs) -> None:

        self.pcd_limit_range = pcd_limit_range
        assert load_type in ('frame_based', 'mv_image_based',
                             'fov_image_based')
        self.load_type = load_type
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            default_cam_key=default_cam_key,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert self.modality is not None
        assert box_type_3d.lower() in ('lidar', 'camera') # TODO: (michbaum) Needs changes if we change the box type

    # TODO: (michbaum) I think this is the main thing we might need to adapt
    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.modality['use_lidar']:
            if 'plane' in info:
                # convert ground plane to velodyne coordinates
                plane = np.array(info['plane'])
                lidar2cam = np.array(
                    info['images']['CAM2']['lidar2cam'], dtype=np.float32)
                reverse = np.linalg.inv(lidar2cam)

                (plane_norm_cam, plane_off_cam) = (plane[:3],
                                                   -plane[:3] * plane[3])
                plane_norm_lidar = \
                    (reverse[:3, :3] @ plane_norm_cam[:, None])[:, 0]
                plane_off_lidar = (
                    reverse[:3, :3] @ plane_off_cam[:, None][:, 0] +
                    reverse[:3, 3])
                plane_lidar = np.zeros_like(plane_norm_lidar, shape=(4, ))
                plane_lidar[:3] = plane_norm_lidar
                plane_lidar[3] = -plane_norm_lidar.T @ plane_off_lidar
            else:
                plane_lidar = None

            info['plane'] = plane_lidar

        if self.load_type == 'fov_image_based' and self.load_eval_anns:
            info['instances'] = info['cam_instances'][self.default_cam_key]

        # (michbaum) This is no longer compatible with our data format, so
        #            we add a custom parser
        # info = super().parse_data_info(info)

        if self.modality['use_lidar']:
            for pc_idx, pc_info in info['lidar_points'].items():
                if 'PC' not in pc_idx:
                    continue
                pc_info['lidar_path'] = \
                    osp.join(
                        self.data_prefix.get('pts', ''),
                        pc_info['lidar_path'])

            info['num_pts_feats'] = info['lidar_points']['num_pts_feats']
            # TODO: (michbaum) Not possible due to the new structure, adapt down the line
            # info['lidar_path'] = info['lidar_points']['lidar_path']

            # (michbaum) We don't work with lidar sweeps, so not adapted
            if 'lidar_sweeps' in info:
                for sweep in info['lidar_sweeps']:
                    file_suffix = sweep['lidar_points']['lidar_path'].split(
                        os.sep)[-1]
                    if 'samples' in sweep['lidar_points']['lidar_path']:
                        sweep['lidar_points']['lidar_path'] = osp.join(
                            self.data_prefix['pts'], file_suffix)
                    else:
                        sweep['lidar_points']['lidar_path'] = osp.join(
                            self.data_prefix['sweeps'], file_suffix)

        if self.modality['use_camera']:
            for cam_id, img_info in info['images'].items():
                if 'img_path' in img_info:
                    if cam_id in self.data_prefix:
                        cam_prefix = self.data_prefix[cam_id]
                    else:
                        cam_prefix = self.data_prefix.get('img', '')
                    img_info['img_path'] = osp.join(cam_prefix,
                                                    img_info['img_path'])
            if self.default_cam_key is not None:
                info['img_path'] = info['images'][
                    self.default_cam_key]['img_path']
                if 'lidar2cam' in info['images'][self.default_cam_key]:
                    info['lidar2cam'] = np.array(
                        info['images'][self.default_cam_key]['lidar2cam'])
                if 'cam2img' in info['images'][self.default_cam_key]:
                    info['cam2img'] = np.array(
                        info['images'][self.default_cam_key]['cam2img'])
                if 'lidar2img' in info['images'][self.default_cam_key]:
                    info['lidar2img'] = np.array(
                        info['images'][self.default_cam_key]['lidar2img'])
                else:
                    info['lidar2img'] = info['cam2img'] @ info['lidar2cam']

        if not self.test_mode:
            # used in training
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = self.parse_ann_info(info)

        return info

    # TODO: (michbaum) Need to bring in the class & instance masks I think?
    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): # TODO: (michbaum) Change to our BBoxType
                  3D ground truth bboxes.
                - bbox_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                  0, 1, 2 represent xxxxx respectively.
        """
        # TODO (michbaum) Not possible, needs adaption
        # ann_info = super().parse_ann_info(info)
        ann_info = None

        # add s or gt prefix for most keys after concat
        # we only process 3d annotations here, the corresponding
        # 2d annotation process is in the `LoadAnnotations3D`
        # in `transforms`
        name_mapping = {
            'bbox_label_3d': 'gt_labels_3d',
            'bbox_label': 'gt_bboxes_labels',
            'bbox': 'gt_bboxes',
            'bbox_3d': 'gt_bboxes_3d',
            'depth': 'depths',
            'center_2d': 'centers_2d',
            'attr_label': 'attr_labels',
            'velocity': 'velocities',
        }
        instances = info['instances']
        # empty gt
        if len(instances) == 0:
            # return None
            pass
        else:
            ann_info = dict()
            ann_info['instances'] = dict()
            # (michbaum) We have an annotation instance for each camera in the scene, so we need
            #            to iterate over them
            # TODO: (michbaum) I think down the line, when we have logic to only train on a subset of
            #       the scene, we need to merge the annotations of the choosen cameras somehow
            for instance_name, instance in instances.items():
                keys = list(instance[0].keys())
                # ann_info = dict()
                ann_info[instance_name] = dict()
                for ann_name in keys:
                    temp_anns = [item[ann_name] for item in instance]
                    # map the original dataset label to training label
                    if 'label' in ann_name and ann_name != 'attr_label':
                        temp_anns = [
                            self.label_mapping[item] for item in temp_anns
                        ]
                    if ann_name in name_mapping:
                        mapped_ann_name = name_mapping[ann_name]
                    else:
                        mapped_ann_name = ann_name

                    if 'label' in ann_name:
                        temp_anns = np.array(temp_anns).astype(np.int64)
                    elif ann_name in name_mapping:
                        temp_anns = np.array(temp_anns).astype(np.float32)
                    else:
                        temp_anns = np.array(temp_anns)

                    ann_info[instance_name][mapped_ann_name] = temp_anns
                ann_info['instances'][instance_name] = info['instances'][instance_name]

                for label in ann_info[instance_name]['gt_labels_3d']:
                    # (michbaum) We count the instances per camera/pointcloud, which is not optimal,
                    #            but not all instances are visible from all camera angles, so I think
                    #            this is better than just counting them once per scene
                    if label != -1:
                        self.num_ins_per_cat[label] += 1

        if ann_info is None:
            ann_info = dict()
            # empty instance
            # TODO: (michbaum) Change to more dimensions when full rotation is in
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

            if self.load_type in ['fov_image_based', 'mv_image_based']:
                ann_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
                ann_info['gt_bboxes_labels'] = np.array(0, dtype=np.int64)
                ann_info['centers_2d'] = np.zeros((0, 2), dtype=np.float32)
                ann_info['depths'] = np.zeros((0), dtype=np.float32)

        # (michbaum) We don't have don't cares at this point so we don't adapt this yet,
        #            but would need to override the function here
        # ann_info = self._remove_dontcare(ann_info)

        # (michbaum) in kitti, lidar2cam = R0_rect @ Tr_velo_to_cam
        lidar2cam = np.array(info['images']['CAM2']['lidar2cam'])
        # convert gt_bboxes_3d to velodyne coordinates with `lidar2cam`
        # TODO: (michbaum) Check if we want it in camera coordinates, otherwise
        #                  change the logic here to iterate over all 3d boxes in the 
        #                  different camera instances of the scene
        # (michbaum) I thiiiink this transforms to lidar/global coors, but we already have that
        # gt_bboxes_3d = CameraInstance3DBoxes(
        #     ann_info['gt_bboxes_3d']).convert_to(self.box_mode_3d,
        #                                          np.linalg.inv(lidar2cam))
        # ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info
