# TODO: (michbaum) Check if necessary
# custom_imports = dict(imports=['projects.example_project.dummy'])
# _base_.model.backbone.type = 'DummyResNet'


_base_ = [
    './extended-kitti-seg.py', '../../../configs/_base_/models/pointnet2_ssg.py',
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


# -------------------------MODEL-------------------------
# (michbaum) pointnet2_ssg.py
# model = dict(
#     type='EncoderDecoder3D',
#     data_preprocessor=dict(type='Det3DDataPreprocessor'),
#     backbone=dict(
#         type='PointNet2SASSG',
#         in_channels=6,  # [xyz, rgb], should be modified with dataset
#         num_points=(1024, 256, 64, 16),
#         radius=(0.1, 0.2, 0.4, 0.8),
#         num_samples=(32, 32, 32, 32),
#         sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256,
#                                                                     512)),
#         fp_channels=(),
#         norm_cfg=dict(type='BN2d'),
#         sa_cfg=dict(
#             type='PointSAModule',
#             pool_mod='max',
#             use_xyz=True,
#             normalize_xyz=False)),
#     decode_head=dict(
#         type='PointNet2Head',
#         fp_channels=((768, 256, 256), (384, 256, 256), (320, 256, 128),
#                      (128, 128, 128, 128)),
#         channels=128,
#         dropout_ratio=0.5,
#         conv_cfg=dict(type='Conv1d'),
#         norm_cfg=dict(type='BN1d'),
#         act_cfg=dict(type='ReLU'),
#         loss_decode=dict(
#             type='mmdet.CrossEntropyLoss',
#             use_sigmoid=False,
#             class_weight=None,  # should be modified with dataset
#             loss_weight=1.0)),
#     # model training and testing settings
#     train_cfg=dict(),
#     test_cfg=dict(mode='slide'))


# model settings
model = dict(
    backbone=dict(
        in_channels=8,  # [xyz, rgb, class_id_prior, instance_id_prior]
    ),
    decode_head=dict(
        num_classes=3,
        ignore_index=3,
        # `class_weight` is generated in data pre-processing, saved in
        # `data/scannet/seg_info/train_label_weight.npy`
        # you can copy paste the values here, or input the file path as
        # `class_weight=data/scannet/seg_info/train_label_weight.npy`
        # loss_decode=dict(class_weight=[
        #     2.389689, 2.7215734, 4.5944676, 4.8543367, 4.096086, 4.907941,
        #     4.690836, 4.512031, 4.623311, 4.9242644, 5.358117, 5.360071,
        #     5.019636, 4.967126, 5.3502126, 5.4023647, 5.4027233, 5.4169416,
        #     5.3954206, 4.6971426
        # ])
        ),
    test_cfg=dict(
        mode='slide', # TODO: (michbaum) Need 'whole' to use whole scene, otherwise will be 'slide'
        num_points=8192,
        block_size=1.5,
        sample_rate=0.5,
        use_normalized_coord=False,
        batch_size=24))

# data settings
train_dataloader = dict(batch_size=16)

# runtime settings
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5),
                     visualization=dict(type='Det3DVisualizationHook', draw=True))
train_cfg = dict(val_interval=5)