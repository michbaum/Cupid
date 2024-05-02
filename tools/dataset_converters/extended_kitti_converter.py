# Adapted code from mmdet3d/tools/data_converter/kitti_converter.py for an extended KITTI dataset

from collections import OrderedDict
from pathlib import Path

import mmcv
import mmengine
import numpy as np
from nuscenes.utils.geometry_utils import view_points

from mmdet3d.structures import points_cam2img
from mmdet3d.structures.ops import box_np_ops
from mmdet3d.structures.bbox_3d import limit_period, rotation_3d_in_axis
from .extended_kitti_data_utils import get_extended_kitti_image_info
from .nuscenes_converter import post_process_coords

# TODO: (michbaum) Maybe add categories for surounding/table/floor 
kitti_categories = ('box')

def _read_metadata_file(path: str) -> dict[str, int]:
    """Read the metadata file in the extended KITTI format.
    It is expected to contain the following:
    num_cameras_per_scene: x
    num_pointclouds_per_scene: y

    Args:
        path (str): Path to the metadata file.

    Returns:
        dict[str, int]: Dictionary containing the metadata.
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    meta = {}
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key, value = line.split(':')
        meta[key] = int(value)
    return meta

def _read_imageset_file(path: str):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

# Currently not used and also not adapted, see _calculate_num_points_in_gt for changes
class _NumPointsInGTCalculater:
    """Calculate the number of points inside the ground truth box. This is the
    parallel version. For the serialized version, please refer to
    `_calculate_num_points_in_gt`.

    Args:
        data_path (str): Path of the data.
        relative_path (bool): Whether to use relative path.
        remove_outside (bool, optional): Whether to remove points which are
            outside of image. Default: True.
        num_features (int, optional): Number of features per point.
            Default: False.
        num_worker (int, optional): the number of parallel workers to use.
            Default: 8.
    """

    def __init__(self,
                 data_path,
                 relative_path,
                 remove_outside=True,
                 num_features=4,
                 num_worker=8) -> None:
        raise NotImplementedError("This functionality is not adapted yet.")
        self.data_path = data_path
        self.relative_path = relative_path
        self.remove_outside = remove_outside
        self.num_features = num_features
        self.num_worker = num_worker

    def calculate_single(self, info):
        pc_info = info['point_cloud']
        image_info = info['image']
        calib = info['calib']
        if self.relative_path:
            v_path = str(Path(self.data_path) / pc_info['lidar_path'])
        else:
            v_path = pc_info['lidar_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32,
            count=-1).reshape([-1, self.num_features])
        rect = calib['R0_rect']
        Trv2c = calib['Tr_velo_to_cam']
        P2 = calib['P2']
        if self.remove_outside:
            points_v = box_np_ops.remove_outside_points(
                points_v, rect, Trv2c, P2, image_info['image_shape'])
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                         axis=1)
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(
            gt_boxes_camera, rect, Trv2c)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)
        return info

    def calculate(self, infos):
        ret_infos = mmengine.track_parallel_progress(self.calculate_single,
                                                     infos, self.num_worker)
        for i, ret_info in enumerate(ret_infos):
            infos[i] = ret_info


# TODO: (michbaum) Adapt completely
def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
                           origin=(0.5, 1.0, 0.5),
                           axis=1):
    """Convert kitti locations, dimensions and angles to corners.

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 3).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 3).
        angles (np.ndarray, optional): Rotation_y in kitti label file with
            shape (N). Defaults to None.
        origin (list or array or float, optional): Origin point relate to
            smallest point. Use (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0)
            in lidar. Defaults to (0.5, 1.0, 0.5).
        axis (int, optional): Rotation axis. 1 for camera and 2 for lidar.
            Defaults to 1.

    Returns:
        np.ndarray: Corners with the shape of (N, 8, 3).
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(lwh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    # TODO: (michbaum) Should still work if format is KITTI based
    corners = box_np_ops.corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        # TODO: (michbaum) Should be able to reuse this function
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
        # TODO: (michbaum) Repeat the function for the other axes and angles!
        # Should work, but need to double check the rotation order and if it's
        # compatible with zyx euler angles
    corners += centers.reshape([-1, 1, 3])
    return corners


# TODO: (michbaum) Adapt completely
def points_in_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0)):
    """Check points in rotated bbox and return indices.

    Note:
        This function is for counterclockwise boxes.

    Args:
        points (np.ndarray, shape=[N, 3+dim]): Points to query.
        rbbox (np.ndarray, shape=[M, 7]): Boxes3d with rotation.
        z_axis (int, optional): Indicate which axis is height.
            Defaults to 2.
        origin (tuple[int], optional): Indicate the position of
            box center. Defaults to (0.5, 0.5, 0).

    Returns:
        np.ndarray, shape=[N, M]: Indices of points in each box.
    """
    # TODO: (michbaum) Needs changes due to more rotation dimensions
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6], origin=origin, axis=z_axis)
    # TODO: (michbaum) These two should still work
    surfaces = box_np_ops.corner_to_surfaces_3d(rbbox_corners)
    indices = box_np_ops.points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices


# TODO: (michbaum) Adapt completely
def camera_to_lidar(points, r_rect, velo2cam):
    """Convert points in camera coordinate to lidar coordinate.

    Note:
        This function is for KITTI only.

    Args:
        points (np.ndarray, shape=[N, 3]): Points in camera coordinate.
        r_rect (np.ndarray, shape=[4, 4]): Matrix to project points in
            specific camera coordinate (e.g. CAM2) to CAM0.
        velo2cam (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinate to lidar coordinate.

    Returns:
        np.ndarray, shape=[N, 3]: Points in lidar coordinate.
    """
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]


# TODO: (michbaum) Adapt completely
def box_camera_to_lidar(data, r_rect, velo2cam):
    """Convert boxes in camera coordinate to lidar coordinate.

    Note:
        This function is for extended KITTI only.

    Args:
        data (np.ndarray, shape=[N, 9]): Boxes in camera coordinate.
        r_rect (np.ndarray, shape=[4, 4]): Matrix to project points in
            specific camera coordinate (e.g. CAM2) to CAM0.
        velo2cam (np.ndarray, shape=[4, 4]): Matrix to project points in
            camera coordinate to lidar coordinate.

    Returns:
        np.ndarray, shape=[N, 9]: Boxes in lidar coordinate.
    """
    # data format is [x, y, z, dim_x, dim_y, dim_z, rot_x, rot_y, rot_z]
    xyz = data[:, :3]
    x_size, y_size, z_size = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    rot_x, rot_y, rot_z = data[:, 6:7], data[:, 7:8], data[:, 8:9]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam) # TODO: (michbaum) Adapt
    # TODO: (michbaum) The rotation needs to be transformed differently
    # Maybe we build the zyx euler angle, from that the rot matrix, multiply
    # the rot matrices and then transform back to euler angles?
    # yaw and dims also needs to be converted
    r_new = -rot_y - np.pi / 2
    r_new = limit_period(r_new, period=np.pi * 2)
    return np.concatenate([xyz_lidar, x_size, z_size, y_size, r_new], axis=1)

# TODO: (michbaum) Adapt completely
#        This is also somewhat duplicated in box_np_ops.py as points_in_rbbox
#        Check if it can be reused
def _calculate_num_points_in_gt(data_path,
                                infos,
                                relative_path,
                                remove_outside=False,
                                num_features=7):
    """
    Calculate the total number of lidar points inside the ground truth bounding boxes. 
    To this end, first always build the complete point cloud for the scene from all 
    partial point clouds.

    Args:
        data_path (str): Source dataset path.
        infos (dict): Dictionary containing the information of the dataset.
        relative_path (bool): Wheter to use relative path.
        remove_outside (bool, optional): Wheter to remove points lying outside the image. Defaults to False.
        num_features (int, optional): Number of channels in the point cloud data. Defaults to 7.
    """
    for info in mmengine.track_iter_progress(infos):
        pc_infos = info['lidar_points']
        image_infos = info['images']
        calib = info['calib']

        # Build the scene point cloud
        scene_points = None
        for pc_info in pc_infos:
            if relative_path:
                v_path = str(Path(data_path) / pc_info['lidar_path'])
            else:
                v_path = pc_info['lidar_path']
            points_v = np.fromfile(
                v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
            if scene_points is None:
                scene_points = points_v
            else:
                # Aggregate the whole point cloud
                scene_points = np.concatenate([scene_points, points_v], axis=0)
            
        # TODO: (michbaum) Exchange with adapted transforms (check extended_kitti_data_utils.py)
        # rect = calib['R0_rect']
        # Trv2c = calib['Tr_velo_to_cam']
        # P2 = calib['P2']

        # By definition in our use case, there are no points outside of the camera view
        # (Since we work with RGB-D cameras)
        if remove_outside:
            raise NotImplementedError("This functionality is not adapted yet.")
            points_v = box_np_ops.remove_outside_points( # TODO: (michbaum) Needs changing
                points_v, rect, Trv2c, P2, image_infos['image_shape'])

        annos = info['annos']
        # For each camera/image in the scene, we calculate the number of lidar points in the ground truth 
        # bounding boxes of the objects in that scene
        img_idx = 0
        for anno in annos:
            # TODO: (michbaum) For the moment, we don't calculate it and
            # just set it to -1, change if needed later

            # num_obj = len([n for n in anno['name'] if n != 'DontCare'])
            num_obj = 0

            # TODO: (michbaum) Uncomment and adapt this if needed
            # dims = anno['dimensions'][:num_obj]
            # loc = anno['location'][:num_obj]
            # rots_x = anno['rotation_x'][:num_obj]
            # rots_y = anno['rotation_y'][:num_obj]
            # rots_z = anno['rotation_z'][:num_obj]
            # # TODO: (michbaum) Careful, changed dimension
            # gt_boxes_camera = np.concatenate([loc, dims, rots_z[..., np.newaxis], 
            #                                 rots_y[..., np.newaxis], rots_x[..., np.newaxis]],
            #                                 axis=1)
            # # Transform the bounding boxes to lidar coordinates
            # gt_boxes_lidar = box_camera_to_lidar( # TODO: (michbaum) Needs changing
            #     gt_boxes_camera, rect, Trv2c)
            # # Determine which points are inside the bounding boxes
            # indices = points_in_rbbox(points_v[:, :3], gt_boxes_lidar) # TODO: (michbaum) Needs changing
            # num_points_in_gt = indices.sum(0)

            # Populate the ignored objects number of points with -1
            num_ignored = len(anno['dimensions']) - num_obj
            num_points_in_gt = np.concatenate(
                [num_points_in_gt, -np.ones([num_ignored])])
            anno['num_points_in_gt'] = num_points_in_gt.astype(np.int32)
        # I think this populates how many points belonging to each object are within a ground 
        # truth bounding box in the point cloud data


# TODO: (michbaum) Adapt to extended KITTI format
def create_extended_kitti_info_file(data_path,
                           pkl_prefix='extended_kitti',
                           with_plane=False,
                           save_path=None,
                           relative_path=True):
    """Create info file of extended KITTI dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Prefix of the info file to be generated.
            Default: 'extended_kitti'.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
        save_path (str, optional): Path to save the info file.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    dataset_metadata: dict[str, int] = _read_metadata_file(str(imageset_folder / 'metadata.txt'))
    num_cameras_per_scene: int = dataset_metadata['num_cameras_per_scene']
    num_pointclouds_per_scene: int = dataset_metadata['num_pointclouds_per_scene']

    # These now contain the scene number used in the splits (6 digits)
    train_scene_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_scene_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    test_scene_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)

    kitti_infos_train = get_extended_kitti_image_info( # Adapted
        path=data_path,
        training=True,
        pointcloud=True,
        calib=True,
        with_plane=with_plane,
        scene_ids=train_scene_ids,
        num_cams_per_scene=num_cameras_per_scene,
        num_pcs_per_scene=num_pointclouds_per_scene,
        relative_path=relative_path)
    # TODO: (michbaum) Currently, calculate_num_points_in_gt is not adapted and
    # just populates -1 for all objects, so bogus
    _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Extended Kitti info train file is saved to {filename}')
    mmengine.dump(kitti_infos_train, filename)

    kitti_infos_val = get_extended_kitti_image_info(
        data_path,
        training=True,
        pointcloud=True,
        calib=True,
        with_plane=with_plane,
        scene_ids=val_scene_ids,
        num_cams_per_scene=num_cameras_per_scene,
        num_pcs_per_scene=num_pointclouds_per_scene,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Extended Kitti info val file is saved to {filename}')
    mmengine.dump(kitti_infos_val, filename)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'Extended Kitti info trainval file is saved to {filename}')
    mmengine.dump(kitti_infos_train + kitti_infos_val, filename)

    kitti_infos_test = get_extended_kitti_image_info(
        data_path,
        training=False,
        labels=False,
        pointcloud=True,
        calib=True,
        with_plane=False,
        scene_ids=test_scene_ids,
        num_cams_per_scene=num_cameras_per_scene,
        num_pcs_per_scene=num_pointclouds_per_scene,
        relative_path=relative_path)
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'Kitti info test file is saved to {filename}')
    mmengine.dump(kitti_infos_test, filename)


# This function is not needed anymore, since our data stems from a RGB-D camera
# and thus inherently contains no data outside of the camera view
def _create_reduced_point_cloud(data_path,
                                info_path,
                                save_path=None,
                                back=False,
                                num_features=4,
                                front_camera_id=2):
    """Create reduced point clouds for given info.

    Args:
        data_path (str): Path of original data.
        info_path (str): Path of data info.
        save_path (str, optional): Path to save reduced point cloud
            data. Default: None.
        back (bool, optional): Whether to flip the points to back.
            Default: False.
        num_features (int, optional): Number of point features. Default: 4.
        front_camera_id (int, optional): The referenced/front camera ID.
            Default: 2.
    """
    kitti_infos = mmengine.load(info_path)

    for info in mmengine.track_iter_progress(kitti_infos):
        pc_info = info['point_cloud']
        image_info = info['image']
        calib = info['calib']

        v_path = pc_info['lidar_path']
        v_path = Path(data_path) / v_path
        points_v = np.fromfile(
            str(v_path), dtype=np.float32,
            count=-1).reshape([-1, num_features])
        rect = calib['R0_rect']
        if front_camera_id == 2:
            P2 = calib['P2']
        else:
            P2 = calib[f'P{str(front_camera_id)}']
        Trv2c = calib['Tr_velo_to_cam']
        # first remove z < 0 points
        # keep = points_v[:, -1] > 0
        # points_v = points_v[keep]
        # then remove outside.
        if back:
            points_v[:, 0] = -points_v[:, 0]
        points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2,
                                                    image_info['image_shape'])
        if save_path is None:
            save_dir = v_path.parent.parent / (v_path.parent.stem + '_reduced')
            if not save_dir.exists():
                save_dir.mkdir()
            save_filename = save_dir / v_path.name
            # save_filename = str(v_path) + '_reduced'
            if back:
                save_filename += '_back'
        else:
            save_filename = str(Path(save_path) / v_path.name)
            if back:
                save_filename += '_back'
        with open(save_filename, 'w') as f:
            points_v.tofile(f)


# This function is not needed anymore, since our data stems from a RGB-D camera
# and thus inherently contains no data outside of the camera view
def create_reduced_point_cloud(data_path,
                               pkl_prefix,
                               train_info_path=None,
                               val_info_path=None,
                               test_info_path=None,
                               save_path=None,
                               with_back=False):
    """Create reduced point clouds for training/validation/testing.

    Args:
        data_path (str): Path of original data.
        pkl_prefix (str): Prefix of info files.
        train_info_path (str, optional): Path of training set info.
            Default: None.
        val_info_path (str, optional): Path of validation set info.
            Default: None.
        test_info_path (str, optional): Path of test set info.
            Default: None.
        save_path (str, optional): Path to save reduced point cloud data.
            Default: None.
        with_back (bool, optional): Whether to flip the points to back.
            Default: False.
    """
    if train_info_path is None:
        train_info_path = Path(data_path) / f'{pkl_prefix}_infos_train.pkl'
    if val_info_path is None:
        val_info_path = Path(data_path) / f'{pkl_prefix}_infos_val.pkl'
    if test_info_path is None:
        test_info_path = Path(data_path) / f'{pkl_prefix}_infos_test.pkl'

    print('create reduced point cloud for training set')
    _create_reduced_point_cloud(data_path, train_info_path, save_path)
    print('create reduced point cloud for validation set')
    _create_reduced_point_cloud(data_path, val_info_path, save_path)
    print('create reduced point cloud for testing set')
    _create_reduced_point_cloud(data_path, test_info_path, save_path)
    if with_back:
        _create_reduced_point_cloud(
            data_path, train_info_path, save_path, back=True)
        _create_reduced_point_cloud(
            data_path, val_info_path, save_path, back=True)
        _create_reduced_point_cloud(
            data_path, test_info_path, save_path, back=True)

# Not used right now
def export_2d_annotation(root_path, info_path, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        mono3d (bool, optional): Whether to export mono3d annotation.
            Default: True.
    """
    # get bbox annotations for camera
    kitti_infos = mmengine.load(info_path)
    cat2Ids = [
        dict(id=kitti_categories.index(cat_name), name=cat_name)
        for cat_name in kitti_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    from os import path as osp
    for info in mmengine.track_iter_progress(kitti_infos):
        coco_infos = get_2d_boxes(info, occluded=[0, 1, 2, 3], mono3d=mono3d)
        (height, width,
         _) = mmcv.imread(osp.join(root_path,
                                   info['image']['image_path'])).shape
        coco_2d_dict['images'].append(
            dict(
                file_name=info['image']['image_path'],
                id=info['image']['image_idx'],
                Tri2v=info['calib']['Tr_imu_to_velo'],
                Trv2c=info['calib']['Tr_velo_to_cam'],
                rect=info['calib']['R0_rect'],
                cam_intrinsic=info['calib']['P2'],
                width=width,
                height=height))
        for coco_info in coco_infos:
            if coco_info is None:
                continue
            # add an empty key for coco format
            coco_info['segmentation'] = []
            coco_info['id'] = coco_ann_id
            coco_2d_dict['annotations'].append(coco_info)
            coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmengine.dump(coco_2d_dict, f'{json_prefix}.coco.json')

# Not used right now
def get_2d_boxes(info, occluded, mono3d=True):
    """Get the 2D annotation records for a given info.

    Args:
        info: Information of the given sample data.
        occluded: Integer (0, 1, 2, 3) indicating occlusion state:
            0 = fully visible, 1 = partly occluded, 2 = largely occluded,
            3 = unknown, -1 = DontCare
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """
    # Get calibration information
    P2 = info['calib']['P2']

    repro_recs = []
    # if no annotations in info (test dataset), then return
    if 'annos' not in info:
        return repro_recs

    # Get all the annotation with the specified visibilties.
    ann_dicts = info['annos']
    mask = [(ocld in occluded) for ocld in ann_dicts['occluded']]
    for k in ann_dicts.keys():
        ann_dicts[k] = ann_dicts[k][mask]

    # convert dict of list to list of dict
    ann_recs = []
    for i in range(len(ann_dicts['occluded'])):
        ann_rec = {}
        for k in ann_dicts.keys():
            ann_rec[k] = ann_dicts[k][i]
        ann_recs.append(ann_rec)

    for ann_idx, ann_rec in enumerate(ann_recs):
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = \
            f"{info['image']['image_idx']}.{ann_idx}"
        ann_rec['sample_data_token'] = info['image']['image_idx']
        sample_data_token = info['image']['image_idx']

        loc = ann_rec['location'][np.newaxis, :]
        dim = ann_rec['dimensions'][np.newaxis, :]
        rot = ann_rec['rotation_y'][np.newaxis, np.newaxis]
        # transform the center from [0.5, 1.0, 0.5] to [0.5, 0.5, 0.5]
        dst = np.array([0.5, 0.5, 0.5])
        src = np.array([0.5, 1.0, 0.5])
        loc = loc + dim * (dst - src)
        offset = (info['calib']['P2'][0, 3] - info['calib']['P0'][0, 3]) \
            / info['calib']['P2'][0, 0]
        loc_3d = np.copy(loc)
        loc_3d[0, 0] += offset
        gt_bbox_3d = np.concatenate([loc, dim, rot], axis=1).astype(np.float32)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box_np_ops.center_to_corner_box3d(
            gt_bbox_3d[:, :3],
            gt_bbox_3d[:, 3:6],
            gt_bbox_3d[:, 6], [0.5, 0.5, 0.5],
            axis=1)
        corners_3d = corners_3d[0].T  # (1, 8, 3) -> (3, 8)
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        camera_intrinsic = P2
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token,
                                    info['image']['image_path'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            repro_rec['bbox_cam3d'] = np.concatenate(
                [loc_3d, dim, rot],
                axis=1).astype(np.float32).squeeze().tolist()
            repro_rec['velo_cam3d'] = -1  # no velocity in KITTI

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            repro_rec['attribute_name'] = -1  # no attribute in KITTI
            repro_rec['attribute_id'] = -1

        repro_recs.append(repro_rec)

    return repro_recs

# Not used right now
def generate_record(ann_rec, x1, y1, x2, y2, sample_data_token, filename):
    """Generate one 2D annotation record given various information on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): file name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, x_size, y_size of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    key_mapping = {
        'name': 'category_name',
        'num_points_in_gt': 'num_lidar_pts',
        'sample_annotation_token': 'sample_annotation_token',
        'sample_data_token': 'sample_data_token',
    }

    for key, value in ann_rec.items():
        if key in key_mapping.keys():
            repro_rec[key_mapping[key]] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in kitti_categories:
        return None
    cat_name = repro_rec['category_name']
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = kitti_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec
