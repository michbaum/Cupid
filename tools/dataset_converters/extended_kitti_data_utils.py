# Adapted from mmdet3d/datasets/kitti_dataset.py for extended KITTI dataset
from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path

import mmengine
import numpy as np
from PIL import Image
from skimage import io

# TODO: (michbaum) Check if all is needed and if there are changes needed

# Since we change the folder structure, we probably need to change all the getters

def get_image_index_str(img_idx: int, use_prefix_id=False) -> str:
    """Returns the image index as a string with a fixed length of 6 or 7 digits, 
    depending on the use of the prefix ID.

    Args:
        img_idx (int): Index of the image/sensor reading.
        use_prefix_id (bool, optional): Wheter to use the prefix_id or not. Defaults to False.

    Returns:
        str: Image index as a string.
    """
    if use_prefix_id:
        return '{:07d}'.format(img_idx)
    else:
        return '{:06d}'.format(img_idx)

def get_scene_index_str(scene_idx: int) -> str:
    """Returns the scene index as a string with a fixed length of 6 digits.

    Args:
        scene_idx (int): Index of the scene.

    Returns:
        str: Scene index as a string.
    """
    return '{:06d}'.format(scene_idx)

def get_extended_kitti_info_path(scene_idx: int,
                                 image_idx: int,
                                 prefix: str,
                                 info_type='images',
                                 file_tail='.png',
                                 training=True,
                                 relative_path=True,
                                 exist_check=True,
                                 use_prefix_id=False) -> str:
    """
    Returns the filepath to a file of a specific type corresponding to the {image_idx}th sensor reading within the 
    {scene_idx}th scene in the extended KITTI dataset.
    
    Args:
        scene_idx (int): Scene index.
        image_idx (int): Image index.
        prefix (str): Prefix path to the dataset.
        info_type (str, optional): Type of information to retrieve. Defaults to 'images'.
        file_tail (str, optional): File extension. Defaults to '.png'.
        training (bool, optional): Whether the data is training or testing data. Defaults to True.
        relative_path (bool, optional): Whether to return the path relative to the prefix. Defaults to True.
        exist_check (bool, optional): Whether to check if the file exists. Defaults to True.
        use_prefix_id (bool, optional): Whether to use the prefix ID. Defaults to False.

    Returns:
        str: Filepath to the file.
    """
    scene_idx_str = get_scene_index_str(scene_idx)
    img_idx_str = get_image_index_str(image_idx, use_prefix_id)
    img_idx_str += file_tail
    prefix = Path(prefix)
    if training:
        file_path = Path('training') / info_type / scene_idx_str / img_idx_str
    else:
        file_path = Path('testing') / info_type / scene_idx_str / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_image_path(scene_idx: int,
                   image_idx: int,
                   prefix: str,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='images',
                   file_tail='.png',
                   use_prefix_id=False) -> str:
    """Returns the path to the image file corresponding to the {image_idx}th sensor reading within the
    {scene_idx}th scene in the extended KITTI dataset.

    Args:
        scene_idx (int): Scene index.
        image_idx (int): Image index.
        prefix (str): Prefix path to the dataset.
        training (bool, optional): Wheter the image is in the training or test split. Defaults to True.
        relative_path (bool, optional): Wheter to return a relative path to the prefix. Defaults to True.
        exist_check (bool, optional): Wheter to check if the file exists. Defaults to True.
        info_type (str, optional): Name of the parent folder of the files to retrieve. Defaults to 'images'.
        file_tail (str, optional): File ending. Defaults to '.png'.
        use_prefix_id (bool, optional): Wheter to utilize the prefix ID in the file names. Defaults to False.

    Returns:
        str: Filepath to the image file.
    """
    return get_extended_kitti_info_path(scene_idx, image_idx, prefix, info_type, file_tail, training,
                               relative_path, exist_check, use_prefix_id)


def get_label_path(scene_idx: int,
                   image_idx: int,
                   prefix: str,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='labels',
                   file_tail='.txt',
                   use_prefix_id=False):
    """
    Returns the path to the label file corresponding to the {image_idx}th sensor reading within the
    {scene_idx}th scene in the extended KITTI dataset.

    Args:
        scene_idx (int): Scene index.
        image_idx (int): Image index.
        prefix (str): Prefix path to the dataset.
        training (bool, optional): Wheter the image is in the training or test split. Defaults to True.
        relative_path (bool, optional): Wheter to return a relative path to the prefix. Defaults to True.
        exist_check (bool, optional): Wheter to check if the file exists. Defaults to True.
        info_type (str, optional): Name of the parent folder of the files to retrieve. Defaults to 'labels'.
        file_tail (str, optional): File ending. Defaults to '.txt'.
        use_prefix_id (bool, optional): Wheter to utilize the prefix ID in the file names. Defaults to False.

    Returns:
        str: Filepath to the label file.
    """
    return get_extended_kitti_info_path(scene_idx, image_idx, prefix, info_type, file_tail, training,
                               relative_path, exist_check, use_prefix_id)


def get_plane_path(scene_idx: int,
                   image_idx: int,
                   prefix: str,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='planes',
                   use_prefix_id=False) -> str:
    """ 
    Returns the path to the plane file corresponding to the {image_idx}th sensor reading within the
    {scene_idx}th scene in the extended KITTI dataset.

    Args:
        scene_idx (int): Scene index.
        image_idx (int): Image index.
        prefix (str): Prefix path to the dataset.
        training (bool, optional): Wheter the image is in the training or test split. Defaults to True.
        relative_path (bool, optional): Wheter to return a relative path to the prefix. Defaults to True.
        exist_check (bool, optional): Wheter to check if the file exists. Defaults to True.
        info_type (str, optional): Name of the parent folder of the files to retrieve. Defaults to 'planes'.
        use_prefix_id (bool, optional): Wheter to utilize the prefix ID in the file names. Defaults to False.

    Returns:
        str: Filepath to the plane file.
    """
    return get_extended_kitti_info_path(scene_idx, image_idx, prefix, info_type, '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_lidar_path(scene_idx: int,
                    image_idx: int,
                    prefix: str,
                    training=True,
                    relative_path=True,
                    exist_check=True,
                    info_type='pointclouds',
                    use_prefix_id=False) -> str:
    """
    Returns the path to the pointcloud file corresponding to the {image_idx}th sensor reading within the
    {scene_idx}th scene in the extended KITTI dataset.

    Args:
        scene_idx (int): Scene index.
        image_idx (int): Image index.
        prefix (str): Prefix path to the dataset.
        training (bool, optional): Wheter the image is in the training or test split. Defaults to True.
        relative_path (bool, optional): Wheter to return a relative path to the prefix. Defaults to True.
        exist_check (bool, optional): Wheter to check if the file exists. Defaults to True.
        info_type (str, optional): Name of the parent folder of the files to retrieve. Defaults to 'pointclouds'.
        use_prefix_id (bool, optional): Wheter to utilize the prefix ID in the file names. Defaults to False.

    Returns:
        str: Filepath to the pointcloud file.
    """
    return get_extended_kitti_info_path(scene_idx, image_idx, prefix, info_type, '.bin', training,
                               relative_path, exist_check, use_prefix_id)


def get_calib_path(scene_idx: int,
                   image_idx: int,
                   prefix: str,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   use_prefix_id=False) -> str:
    """ 
    Returns the path to the calibration file corresponding to the {image_idx}th sensor reading within the
    {scene_idx}th scene in the extended KITTI dataset.

    Args:
        scene_idx (int): Scene index.
        image_idx (int): Image index.
        prefix (str): Prefix path to the dataset.
        training (bool, optional): Wheter the image is in the training or test split. Defaults to True.
        relative_path (bool, optional): Wheter to return a relative path to the prefix. Defaults to True.
        exist_check (bool, optional): Wheter to check if the file exists. Defaults to True.
        use_prefix_id (bool, optional): Wheter to utilize the prefix ID in the file names. Defaults to False.

    Returns:
        str: Filepath to the calibration file.
    """
    return get_extended_kitti_info_path(scene_idx, image_idx, prefix, 'calib', '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_pose_path(scene_idx: int,
                  image_idx: int,
                  prefix: str,
                  training=True,
                  relative_path=True,
                  exist_check=True,
                  use_prefix_id=False):
    """ 
    Returns the path to the pose file corresponding to the {image_idx}th sensor reading within the
    {scene_idx}th scene in the extended KITTI dataset.

    Args:
        scene_idx (int): Scene index.
        image_idx (int): Image index.
        prefix (str): Prefix path to the dataset.
        training (bool, optional): Wheter the image is in the training or test split. Defaults to True.
        relative_path (bool, optional): Wheter to return a relative path to the prefix. Defaults to True.
        exist_check (bool, optional): Wheter to check if the file exists. Defaults to True.
        use_prefix_id (bool, optional): Wheter to utilize the prefix ID in the file names. Defaults to False.

    Returns:
        str: Filepath to the pose file.
    """
    return get_extended_kitti_info_path(scene_idx, image_idx, prefix, 'pose', '.txt', training,
                               relative_path, exist_check, use_prefix_id)


# TODO: (michbaum) Check if this is needed
def get_timestamp_path(scene_idx: int,
                       image_idx: int,
                       prefix: str,
                       training=True,
                       relative_path=True,
                       exist_check=True,
                       use_prefix_id=False):
    """ 
    Returns the path to the timestamp file corresponding to the {image_idx}th sensor reading within the
    {scene_idx}th scene in the extended KITTI dataset.

    Args:
        scene_idx (int): Scene index.
        image_idx (int): Image index.
        prefix (str): Prefix path to the dataset.
        training (bool, optional): Wheter the image is in the training or test split. Defaults to True.
        relative_path (bool, optional): Wheter to return a relative path to the prefix. Defaults to True.
        exist_check (bool, optional): Wheter to check if the file exists. Defaults to True.
        use_prefix_id (bool, optional): Wheter to utilize the prefix ID in the file names. Defaults to False.

    Returns:
        str: Filepath to the timestamp file.
    """
    return get_extended_kitti_info_path(scene_idx, image_idx, prefix, 'timestamp', '.txt', training,
                               relative_path, exist_check, use_prefix_id)


def get_label_anno(label_path: str) -> dict[str, str|np.ndarray]:
    """
    Extracts the label annotations from a label file in the extended KITTI format.
    Expects the object rotation to be in zyx Euler angle format.

    Args:
        label_path (str): Path to the label file.

    Returns:
        dict[str, str|np.ndarray]: Converted label annotations with the following keys:
            - name: Object name.
            - truncated: Truncation of the object.
            - occluded: Occlusion of the object.
            - alpha: Observation angle of the object.
            - bbox: Bounding box of the object.
            - dimensions: Dimensions of the object.
            - location: Location of the object.
            - rotation_x: Rotation around the x-axis.
            - rotation_y: Rotation around the y-axis.
            - rotation_z: Rotation around the z-axis.
    """
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_x': [],
        'rotation_y': [],
        'rotation_z': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare']) # TODO: (michbaum) Could be removed
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                    for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    # TODO: (michbaum) Check that this is consistent between export and import
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_x'] = np.array([float(x[14])
                                          for x in content]).reshape(-1)
    annotations['rotation_y'] = np.array([float(x[15])
                                          for x in content]).reshape(-1)
    annotations['rotation_z'] = np.array([float(x[16])
                                          for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 18:  # have score
        annotations['score'] = np.array([float(x[17]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    # We don't consider multipart objects, so we just give every object a unique group id
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


def _extend_intrinsic_matrix(mat):
    mat = np.concatenate([mat, np.array([[0.], [0.], [0.]])], axis=1)
    return mat

def _extend_extrinsic_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_extended_kitti_image_info(path: str,
                         training=True,
                         labels=True,
                         pointcloud=True,
                         calib=False,
                         with_plane=False,
                         scene_ids=100, # Overwritten by train/val/test split containing scene folders
                         num_cams_per_scene=20, # Number of images per scene, overwritten by metadata
                         num_pcs_per_scene=20, # Number of pointclouds per scene, overwritten by metadata
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True) -> dict:
    # TODO: (michbaum) Add doc
    """

    Extended KITTI annotation format version 1:
    {
        scene_idx: ...
        images: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_clouds: {
            num_features: 7 (x, y, z, r, g, b, label)
            lidar_paths: ...
        }
        calib: {
            K0: ...
            K1: ...
            ...
            Tr_lidar_to_cam: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_x: [num_gt] angle array, expected to be zyx Euler angles
            rotation_y: [num_gt] angle array, expected to be zyx Euler angles
            rotation_z: [num_gt] angle array, expected to be zyx Euler angles
            name: [num_gt] ground truth name array
            difficulty: kitti difficulty
            group_ids: used for multi-part object
        }
    }

    Args:
        path (str): Path to the dataset.
        training (bool, optional): Whether the data is training or testing data. Defaults to True.
        labels (bool, optional): Whether to include the label annotations. Defaults to True.
        pointcloud (bool, optional): Whether to include the pointcloud information. Defaults to True.
        calib (bool, optional): Whether to include the calibration information. Defaults to False.
        with_plane (bool, optional): Whether to include the plane information. Defaults to False.
        scene_ids (int, optional): Number of scenes in the dataset to include. Defaults to 100.
        num_cams_per_scene (int, optional): Number of cameras per scene. Defaults to 20.
        num_pcs_per_scene (int, optional): Number of pointclouds per scene. Defaults to 20.
        extend_matrix (bool, optional): Whether to extend the intrinsic and extrinsic matrices. Defaults to True.
        num_worker (int, optional): Number of workers for parallel processing. Defaults to 8.
        relative_path (bool, optional): Whether to return relative paths. Defaults to True.
        with_imageshape (bool, optional): Whether to include the image shape. Defaults to True.
    
    Returns:
        dict: Information dictionary containing the annotations.

    """
    root_path = Path(path)
    if not isinstance(scene_ids, list):
        scene_ids = list(range(scene_ids))

    image_ids = list(range(num_cams_per_scene))
    pc_ids = list(range(num_pcs_per_scene))

    def map_func(scene_idx):
        # Same as before, we build a single pickle file for a whole "scene", but now we have
        # a variable number of cameras and pointclouds per file and need to handle that
        info = {'scene_idx': scene_idx}
        image_info = {}
        label_info = {}
        pc_info = {'num_features': 7} # (x, y, z, r, g, b, label)
        calib_info = {}
        # Gather all the image paths and information
        for image_idx in image_ids:
            image_info_i = {'image_idx': image_idx}
            image_info_i['image_path'] = get_image_path(
                scene_idx, image_idx, path, training, relative_path)
            if with_imageshape:
                img_path = image_info_i['image_path']
                if relative_path:
                    img_path = str(root_path / img_path)
                image_info_i['image_shape'] = np.array(
                    io.imread(img_path).shape[:2], dtype=np.int32)
            if labels:
                label_path = get_label_path(scene_idx, image_idx, path, training, relative_path)
                if relative_path:
                    label_path = str(root_path / label_path)
                label_info[f'CAM{image_idx}'] = get_label_anno(label_path)
            image_info[f'CAM{image_idx}'] = image_info_i

        info['images'] = image_info
        if labels:
            info['annos'] = label_info
            # TODO: (michbaum) Check if this is correct
            add_difficulty_to_annos(info)

        # Gather all the point cloud paths and information
        if pointcloud: 
            for pc_idx in pc_ids:
                pc_info_i = {'pc_idx': pc_idx}
                pc_info_i['lidar_path'] = get_lidar_path(
                    scene_idx, image_idx, path, training, relative_path)
                pc_info[f'PC{pc_idx}'] = pc_info_i
        info['lidar_points'] = pc_info

        # Gather all the calibration information
        # Contrary to the classical KITTI format, we don't have stereo images, but multiple cameras
        # in a general setup. This means we have to read in the individual intrinsics and extrinsics
        # here and then later process them in the update_infos_to_v2.py to get all necessary
        # transformations.
        if calib:
            calib_path = get_calib_path( 
                scene_idx, image_idx, path, training, relative_path)
            if relative_path:
                calib_path = str(root_path / calib_path)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            calib_info = {}
            camera_extrinsics = []
            # We have one intrinsic matrix line per camera in the scene followed by a line containing its
            # extrinsic matrix, whereas all the pointclouds are expected to be 
            # in world frame already (to be compatible regardless of the number of pointcloud sources)
            for cam_id in range(num_cams_per_scene):
                # Extract the 9 camera intrinsic parameters
                K = np.array([float(info) for info in lines[2 * cam_id].split(' ')[1:10]
                            ]).reshape([3, 3])
                if extend_matrix:
                    # Extend to a (3, 4) matrix
                    K = _extend_intrinsic_matrix(K)
                # Extract the 12 camera extrinsic parameters
                Tr_imu_to_cam = np.array([float(info) for info in lines[2 * cam_id + 1].split(' ')[1:13]
                                        ]).reshape([3, 4])
                if extend_matrix:
                    # Extend to a (4, 4) matrix
                    Tr_imu_to_cam = _extend_extrinsic_matrix(Tr_imu_to_cam)
                calib_info[f'K{cam_id}'] = K
                calib_info[f'Tr_imu_to_cam{cam_id}'] = Tr_imu_to_cam
            
            # In our use case, these will be unitary matrices, but we allow for a discrepancy
            Tr_imu_to_lidar = np.array([
                float(info) for info in lines[2 * num_cams_per_scene].split(' ')[1:13]
            ]).reshape([3, 4])
            if extend_matrix:
                Tr_imu_to_lidar = _extend_extrinsic_matrix(Tr_imu_to_lidar)
            
            calib_info['Tr_imu_to_lidar'] = Tr_imu_to_lidar
            info['calib'] = calib_info

        if with_plane: # TODO: (michbaum) Could add plane information, idk if needed/beneficial
            raise NotImplementedError('We don\'t support plane information yet.')
            plane_path = get_plane_path(image_idx, path, training, relative_path)
            if relative_path:
                plane_path = str(root_path / plane_path)
            lines = mmengine.list_from_file(plane_path)
            info['plane'] = np.array([float(i) for i in lines[3].split()])

        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, scene_ids)

    return list(image_infos)


def kitti_anno_to_label_file(annos, folder):
    folder = Path(folder)
    for anno in annos:
        image_idx = anno['metadata']['image_idx']
        label_lines = []
        for j in range(anno['bbox'].shape[0]):
            label_dict = {
                'name': anno['name'][j],
                'alpha': anno['alpha'][j],
                'bbox': anno['bbox'][j],
                'location': anno['location'][j],
                'dimensions': anno['dimensions'][j],
                'rotation_y': anno['rotation_y'][j],
                'score': anno['score'][j],
            }
            label_line = kitti_result_line(label_dict)
            label_lines.append(label_line)
        label_file = folder / f'{get_image_index_str(image_idx)}.txt'
        label_str = '\n'.join(label_lines)
        with open(label_file, 'w') as f:
            f.write(label_str)


def add_difficulty_to_annos(info: dict[str, str|np.ndarray]) -> None:
    """
    Update the annotations with the difficulty level of the objects based
    on their occlusion, truncation and height.

    Args:
        info (dict[str, str | np.ndarray]): Information dictionary containing the annotations.
    """
    # TODO: (michbaum) Might need changing for our use-case
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation

    # We now have multiple annotations - one for each camera
    for anno in info['annos']:
        dims = anno['dimensions']  # lhw format
        bbox = anno['bbox']
        height = bbox[:, 3] - bbox[:, 1]
        occlusion = anno['occluded']
        truncation = anno['truncated']
        diff = []
        easy_mask = np.ones((len(dims), ), dtype=bool)
        moderate_mask = np.ones((len(dims), ), dtype=bool)
        hard_mask = np.ones((len(dims), ), dtype=bool)
        i = 0
        for h, o, t in zip(height, occlusion, truncation):
            if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
                easy_mask[i] = False
            if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
                moderate_mask[i] = False
            if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
                hard_mask[i] = False
            i += 1
        is_easy = easy_mask
        is_moderate = np.logical_xor(easy_mask, moderate_mask)
        is_hard = np.logical_xor(hard_mask, moderate_mask)

        for i in range(len(dims)):
            if is_easy[i]:
                diff.append(0)
            elif is_moderate[i]:
                diff.append(1)
            elif is_hard[i]:
                diff.append(2)
            else:
                diff.append(-1)
        anno['difficulty'] = np.array(diff, np.int32)
    


def kitti_result_line(result_dict, precision=4):
    prec_float = '{' + ':.{}f'.format(precision) + '}'
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError('you must specify a value for {}'.format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError('unknown key. supported key:{}'.format(
                res_dict.keys()))
    return ' '.join(res_line)
