# TODO: (michbaum) Check if necessary
# custom_imports = dict(imports=['projects.example_project.dummy'])
# _base_.model.backbone.type = 'DummyResNet'


_base_ = [
    './extended-kitti-seg.py',
    '../../../configs/_base_/schedules/seg-cosine-200e.py', 
    '../../../configs/_base_/default_runtime.py'
]

# -------------------------DEFAULT RUNTIME-------------------------
# default_scope = 'mmdet3d'

# default_hooks = dict(
#     timer=dict(type='IterTimerHook'),
#     logger=dict(type='LoggerHook', interval=50),
#     param_scheduler=dict(type='ParamSchedulerHook'),
#     checkpoint=dict(type='CheckpointHook', interval=-1),
#     sampler_seed=dict(type='DistSamplerSeedHook'),
#     visualization=dict(type='Det3DVisualizationHook'))

# env_cfg = dict(
#     cudnn_benchmark=False,
#     mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
#     dist_cfg=dict(backend='nccl'),
# )

# log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# log_level = 'INFO'
# load_from = None
# resume = False


# -------------------------OPTIMIZER-------------------------
# # (michbaum) seg-cosine-50e
# # This schedule is mainly used on S3DIS dataset in segmentation task
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='Adam', lr=0.001, weight_decay=0.001),
#     clip_grad=None)

# param_scheduler = [
#     dict(
#         type='CosineAnnealingLR',
#         T_max=50,
#         eta_min=1e-5,
#         by_epoch=True,
#         begin=0,
#         end=50)
# ]

# # runtime settings
# train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
# val_cfg = dict()
# test_cfg = dict()

# # Default setting for scaling LR automatically
# #   - `enable` means enable scaling LR automatically
# #       or not by default.
# #   - `base_batch_size` = (2 GPUs) x (16 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=32)

# ------------------------PARAMETERS---------------------
clustering_method = 'kmeans'  # (michbaum) 'kmeans' or 'meanshift', meanshift is slower
kmeans_clusters = 20
meanshift_bandwidth = 0.6  # (michbaum) From https://github.com/prs-eth/PanopticSegForMobileMappingPointClouds/blob/main/conf/models/panoptic/pointnet2.yaml#L168 
segmentation_dropout = 0.5
instance_dropout = 0.
embed_loss_weight = 0.5  # (michbaum) 0.1 originally from paper: https://arxiv.org/pdf/2304.13980
# ------------------------PARAMETERS---------------------


# -------------------------MODEL-------------------------
# (michbaum) pointnet2_ssg.py
model = dict(
    type='CUPIDPanoptic',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(
        type='PointNet2SASSG',
        in_channels=8,  # [xyz, rgb, class_prior, instance_prior], should be modified with dataset
        num_points=(1024, 256, 64, 16),
        radius=(0.1, 0.2, 0.4, 0.8),
        num_samples=(32, 32, 32, 32),
        sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256,
                                                                    512)),
        fp_channels=(),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=False)),
    decode_head=dict(
        type='PointNet2PanopticHead',
        num_classes=3,
        embed_dim=5,
        clustering_method=clustering_method, # (michbaum) or 'meanshift', meanshift is slower
        kmeans_clusters=kmeans_clusters,
        meanshift_bandwidth=meanshift_bandwidth, 
        ignore_index=0, # (michbaum) Need be the same as in the dataset config
        fp_channels=((768, 256, 256), (384, 256, 256), (320, 256, 128),
                     (128, 128, 128, 128)),
        channels=128,
        dropout_ratio=segmentation_dropout,
        embed_dropout_ratio=instance_dropout,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,  # should be modified with dataset
            loss_weight=1.0),
        ins_loss_decode=dict(
            type='DiscriminativeLoss',
            loss_weight=embed_loss_weight), # (michbaum) Weighting from paper: https://arxiv.org/pdf/2304.13980
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(
        mode='slide', # TODO: (michbaum) Need 'whole' to use whole scene, otherwise will be 'slide'
        num_points=8192,
        block_size=1.5,
        sample_rate=0.5,
        use_normalized_coord=False,
        batch_size=24))

# data settings
train_dataloader = dict(batch_size=16)

val_evaluator = dict(type='CUPIDPanopticMetric')
test_evaluator = val_evaluator

# runtime settings
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10),
                    # TODO: (michbaum) Would need changes in the Pack3DDet function to save the correct
                    #       lidar path & maybe even more changes for the visualization
                    #  visualization=dict(type='Det3DVisualizationHook', draw=True, vis_task='lidar_seg', show=True, wait_time=0.01)
                     )
train_cfg = dict(val_interval=10)

# load_from = 'work_dirs/pointnet2_seg_only/pointnet++_seg_1000_train_w_class_priors/epoch_200.pth'
# resume = True