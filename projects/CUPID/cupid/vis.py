import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer

color_map = {}

def get_instance_color(instance_id):
    if instance_id not in color_map:
        color_map[instance_id] = np.random.rand(3) * 255
    return color_map[instance_id]

def get_color(class_id, instance_id):
    if class_id == 0:
        return 255, 255, 255
    elif class_id == 1:
        return 20, 20, 20
    elif class_id == 2:
        return get_instance_color(instance_id)
    else:
        raise ValueError(f'Invalid class_id: {class_id}')


points = np.fromfile('/home/michael/master_thesis/code/robin2/learning/datasets/mmdet3d/10_scns_3_cams/training/pointclouds/000000/000000.bin', dtype=np.float32)
points = points.reshape(-1, 8)
visualizer = Det3DLocalVisualizer()
mask = np.asarray([[r, g, b] for r, g, b in [get_color(point[-2], point[-1]) for point in points]])
points_with_mask = np.concatenate((points[:, :3], mask), axis=-1)
# Draw 3D points with mask
visualizer.set_points(points, pcd_mode=2, vis_mode='add', mode='xyzrgb')
visualizer.draw_seg_mask(points_with_mask)
visualizer.show()