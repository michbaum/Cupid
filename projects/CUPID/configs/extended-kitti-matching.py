# For Extended KITTI seg we usually do a binary classification on instance pointcloud pairs
# This is the base config for models training on the Extended KITTI dataset for matching tasks
# -> overwrite stuff you want to change in the downstream model config, not here

class_names = ('background', 'table', 'box')
point_cloud_range = [-3, -3, -0.5, 3, 3, 1] # TODO: (michbaum) Change this if necessary
metainfo = dict(classes=class_names)
dataset_type = 'ExtendedKittiSegDataset'
# TODO: (michbaum) Change accordingly
# data_root = 'data/extended_kitti/1000_scns_5_cams_reshuffled/'
data_root = 'data/extended_kitti/50_scns_5_cams_reshuffled/'
# data_root = 'data/extended_kitti/10_scns_3_cams_reshuffled/' 
input_modality = dict(use_lidar=True, use_camera=False)
train_data_prefix = dict(
    pts='training/pointclouds',
    pts_instance_mask='training/labels/',
    pts_semantic_mask='training/labels/')
test_data_prefix = dict(
    pts='testing/pointclouds',
    pts_instance_mask='testing/labels/',
    pts_semantic_mask='testing/labels/')

backend_args = None

# -----------------------------------DATA PREPARATION-----------------------------------

# PARAMETERS
num_points = 8192 # (michbaum) Change this to train a model on more sampled input points per instance mask
# num_points = 2048 # (michbaum) Change this to train a model on more sampled input points per instance mask
min_points_per_instance = 400 # (michbaum) Minimum size of fragmented instance pointcloud to be considered in matching
num_views_used = 2 # (michbaum) Change this to train a model for more cameras in the scene
num_views_used_eval = [1, 3] # (michbaum) Make sampling deterministic in eval, adjacent cameras have ~60° angle
max_supported_instances_per_scene = 18 # (michbaum) Change this to train a model on more object instances per scene
matching_instance_class = 2 # (michbaum) The class index of the instances we want to match
pc_dimensions_used = [0, 1, 2, 3, 4, 5, 6, 7] # (michbaum) Change this to use more dimensions of the pointcloud
pc_dims_used_encoder = 8 # (michbaum) How many dimensions to use in the encoder -> 8 seems better
# pc_dimensions_used = [0, 1, 2, 3, 4, 5] # (michbaum) w/o priors
# ~ PARAMETERS

train_pipeline = [
    dict(
        type='LoadEKittiPointsFromFile',
        coord_type='LIDAR',
        shift_height=False,
        use_color=True,
        use_prior_labels=True if len(pc_dimensions_used) > 6 else False,
        load_dim=8,
        use_dim=pc_dimensions_used,
        backend_args=backend_args),
    dict(
        type='LoadEKittiAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=False,
        with_panoptic_3d=True, 
        backend_args=backend_args),
    # (michbaum) Sample and combine n pointclouds per scene here producing more samples
    dict(
        type='SampleKViewsFromScene', 
        num_views=num_views_used,
    ),
    # (michbaum) Filter out points that are not in the point_cloud_range -> ROI of the table & boxes
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),

    # (michbaum) Maps class labels newly if needed, depending on the ignore idx etc.
    dict(type='EKittiPointSegClassMapping'),

    # TODO: (michbaum) Probably want to summarize points here (concatenating the labels they have)
    #                  Maybe via a voxelization step, only keeping one (random) point per voxel with
    #                  all the corresponding labels -> also needs to summarize the annotations
    # dict(
    #     type='IndoorPatchPointSample',
    #     num_points=num_points,
    #     block_size=1.5,
    #     ignore_index=len(class_names),
    #     use_normalized_coord=False,
    #     enlarge_size=0.2,
    #     min_unique_num=None),
    #
    # (michbaum) Normalizes color to [0, 1] -> Does NOT compute mean color in pointcloud or something
    dict(type='NormalizePointsColor', color_mean=None),
    # (michbaum) Randomly negates x or y coordinate of points to generate new scenes -> More train data is good
    # TODO: (michbaum) Put back in
    # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
    # (michbaum) Should rotate, scale and translate the pointcloud -> Again more train data
    #            Also, since the table always has the same rotation in our simulation data, rotation
    #            augmentation is probably necessary to guarantee generalization
    #            - Scaling is from the origin, and I think it would be nice to generalize to other
    #              scales of tables and boxes (we have 9 fixed box types otherwise)
    # TODO: (michbaum) Put back in 
    # dict(type='GlobalRotScaleTrans',
    #      rot_range=[-1.5708, 1.5708],
    #      scale_ratio_range=[0.95, 1.05]),
    # dict(type='PointShuffle'), # (michbaum) Shuffle points in the pointcloud -> GREATLY DETERIORATES PERFORMANCE
    # dict(type='RandomJitterPoints'), # TODO: (michbaum) Could be interesting for us to close real-sim gap

    # (michbaum) Novel instance based sampling -> samples num_points points from each instance and puts it in different channels
    #            Also populates the annotations with the correct instance -> gt_instance mapping for training and evaluation
    dict(type='PreProcessInstanceMatching',
         num_points=num_points,
         min_points_per_instance=min_points_per_instance,
         relevant_class_idx=matching_instance_class,
         num_views_used=num_views_used,
         max_instances=max_supported_instances_per_scene,
         pc_dims_used=pc_dims_used_encoder),

    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask', 'pts_instance_mask', 'instance_gt_mapping', 'pcd_to_instance_mapping'])
]
eval_pipeline = [
    dict(
        type='LoadEKittiPointsFromFile',
        coord_type='LIDAR',
        shift_height=False,
        use_color=True,
        use_prior_labels=True if len(pc_dimensions_used) > 6 else False,
        load_dim=8,
        use_dim=pc_dimensions_used,
        backend_args=backend_args),
    dict(
        type='LoadEKittiAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=False,
        with_panoptic_3d=True,
        backend_args=backend_args),

    # (michbaum) Sample and combine n pointclouds per scene here producing more samples
    dict(
        type='SampleKViewsFromScene',
        num_views=num_views_used_eval,
    ),
    # (michbaum) Filter out points that are not in the point_cloud_range -> ROI of the table & boxes
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),


    # (michbaum) Maps class labels newly if needed, depending on the ignore idx etc.
    dict(type='EKittiPointSegClassMapping'), # TODO: (michbaum) Originally not here, don't know why

    dict(type='NormalizePointsColor', color_mean=None),
    # dict(type='PointShuffle'), # (michbaum) Again, great performance deterioration

    # (michbaum) Novel instance based sampling -> samples num_points points from each instance and puts it in different channels
    #            Also populates the annotations with the correct instance -> gt_instance mapping for training and evaluation
    dict(type='PreProcessInstanceMatching',
         num_points=num_points,
         min_points_per_instance=min_points_per_instance,
         relevant_class_idx=matching_instance_class,
         num_views_used=num_views_used,
         max_instances=max_supported_instances_per_scene,
         pc_dims_used=pc_dims_used_encoder),
    dict(type='Pack3DDetInputs', keys=['points'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
# we need to load gt seg_mask!
test_pipeline = [
    dict(
        type='LoadEKittiPointsFromFile',
        coord_type='LIDAR',
        shift_height=False,
        use_color=True,
        use_prior_labels=True if len(pc_dimensions_used) > 6 else False,
        load_dim=8,
        use_dim=pc_dimensions_used,
        backend_args=backend_args),
    dict(
        type='LoadEKittiAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=False,
        with_panoptic_3d=True,
        backend_args=backend_args),

    # (michbaum) Sample and combine n pointclouds per scene here producing more samples
    dict(
        type='SampleKViewsFromScene', 
        num_views=num_views_used_eval,
    ),
    # (michbaum) Filter out points that are not in the point_cloud_range -> ROI of the table & boxes
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),


    # (michbaum) Maps class labels newly if needed, depending on the ignore idx etc.
    dict(type='EKittiPointSegClassMapping'), # TODO: (michbaum) Originally not here, don't know why

    dict(type='NormalizePointsColor', color_mean=None),
    # dict(type='PointShuffle'), # (michbaum) Same as above

    # TODO: (michbaum) Whether to randomly jitter the pointcloud during evaluation to make it more realistic
    # dict(type='RandomJitterPoints',
    #      jitter_std=[0.01, 0.01, 0.01],
    #     #  clip_range=[-0.01, 0.01],
    #      clip_range=[-0.05, 0.05],
    #      ),

    # (michbaum) Novel instance based sampling -> samples num_points points from each instance and puts it in different channels
    #            Also populates the annotations with the correct instance -> gt_instance mapping for training and evaluation
    dict(type='PreProcessInstanceMatching',
         num_points=num_points,
         min_points_per_instance=min_points_per_instance,
         relevant_class_idx=matching_instance_class,
         num_views_used=num_views_used,
         max_instances=max_supported_instances_per_scene,
         pc_dims_used=pc_dims_used_encoder),
         
    dict(type='Pack3DDetInputs', keys=['points'])
]
tta_pipeline = [ # (michbaum) Test-Time Augmentation pipeline -> Not sure if we need this
    dict(
        type='LoadEKittiPointsFromFile',
        coord_type='LIDAR',
        shift_height=False,
        use_color=True,
        use_prior_labels=True if len(pc_dimensions_used) > 6 else False,
        load_dim=8,
        use_dim=pc_dimensions_used,
        backend_args=backend_args),
    dict(
        type='LoadEKittiAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=False,
        with_panoptic_3d=True,
        backend_args=backend_args),

    # (michbaum) Sample and combine n pointclouds per scene here producing more samples
    dict(
        type='SampleKViewsFromScene', 
        num_views=num_views_used_eval,
    ),
    # (michbaum) Filter out points that are not in the point_cloud_range -> ROI of the table & boxes
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),

    # TODO: (michbaum) Check if the seg label mapping really can be left out here?

    dict(type='NormalizePointsColor', color_mean=None),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(
                type='RandomFlip3D', # TODO: (michbaum) Does nothing with these values - why here?
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=0.)
        ], [dict(type='Pack3DDetInputs', keys=['points'])]])
]

# -----------------------------------DATA LOADERS---------------------------------
train_dataloader = dict(
    batch_size=8, # TODO: (michbaum) Change accordingly - also is overwritten in the model config, prefer changing there
    num_workers=4, # TODO: (michbaum) Change accordingly - also is overwritten in the model config, prefer changing there
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='extended_kitti_infos_train.pkl',
        metainfo=metainfo,
        data_prefix=train_data_prefix,
        pipeline=train_pipeline,
        modality=input_modality,
        ignore_index=len(class_names), # TODO: (michbaum) Last class is the ignore index -> Check that we use this correctly (I think we do by adding a -1 class idx)
        test_mode=False,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1, 
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='extended_kitti_infos_val.pkl',
        # ann_file='extended_kitti_infos_train.pkl', # (michbaum) Sanity check
        metainfo=metainfo,
        data_prefix=train_data_prefix,
        pipeline=eval_pipeline,
        modality=input_modality,
        ignore_index=len(class_names), # TODO: (michbaum) Last class is the ignore index -> Check that we use this correctly (I think we do by adding a -1 class idx)
        test_mode=True, # (michbaum) This needs to be True since we want to get the performance on the val set
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='extended_kitti_infos_test.pkl',
        metainfo=metainfo,
        data_prefix=test_data_prefix,
        pipeline=test_pipeline,
        modality=input_modality,
        ignore_index=len(class_names),
        scene_idxs=data_root + 'ImageSets/test.txt',
        test_mode=True,
        backend_args=backend_args))

val_evaluator = dict(type='MatchMetric')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend'), 
                dict(type='WandbVisBackend', # TODO: (michbaum) Probably needs other args -> want to log train vs. eval performance for example
                init_kwargs={
                    'project': 'master_thesis'
                })
                ]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

tta_model = dict(type='Seg3DTTAModel') # TODO: (michbaum) Don't really know what this does - seems to just "interpret" the segmetnation results
