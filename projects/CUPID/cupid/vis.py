import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer

points = np.fromfile('/home/michael/master_thesis/code/robin2/learning/datasets/mmdet3d/10_scns_3_cams/training/pointclouds/000000/000000.bin', dtype=np.float32)
points = points.reshape(-1, 8)[:, :3]
visualizer = Det3DLocalVisualizer()
mask = np.random.rand(points.shape[0], 3)
points_with_mask = np.concatenate((points, mask), axis=-1)
# Draw 3D points with mask
visualizer.set_points(points, pcd_mode=2, vis_mode='add')
visualizer.draw_seg_mask(points_with_mask)
visualizer.show()