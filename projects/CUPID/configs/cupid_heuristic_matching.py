_base_ = [
    './extended-kitti-matching.py',
    '../../../configs/_base_/schedules/seg-cosine-200e.py', # TODO: (michbaum) Change to set the epochs
    '../../../configs/_base_/default_runtime.py'
]

custom_imports = dict(imports=['projects.CUPID.cupid'])

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

# -----------------------PARAMETERS----------------------
matching_range = None # (michbaum) Euclidean range within feature vectors of the instance masks will be matched. Set None for none.
use_xyz = False # (michbaum) Whether to use clustered feature xyz as a part of features
labels = ('match', 'no_match')
balance_classes = True # (michbaum) Whether we sample equal amount of match and no_match instances in training per scene
# -----------------------PARAMETERS----------------------

# -----------------HEURISTIC PARAMETERS------------------
near_point_threshold = 0.005
min_number_near_points = 50
postprocess_matches=True # (michbaum) Whether to postprocess matching results (e.g. only allow one match per instance mask)
postprocess_strategy='greedy' # (michbaum) 'greedy' or 'hungarian' for optimal postprocessing of matches using the Hungarian algorithm 
visualize_fails = False # (michbaum) Whether to visualize the failed matches (false positives and false negatives)
# -----------------HEURISTIC PARAMETERS------------------

# -------------------------MODEL-------------------------
# (michbaum) CUPID
model = dict(
    type='CUPIDMatching',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    
    postprocess_matches=postprocess_matches,
    postprocess_strategy=postprocess_strategy,
    visualize_fails=visualize_fails,

    backbone=dict(
        type='PointNet2SASSG',
        in_channels=8,  # [xyz, rgb, class_id_prior, instance_id_prior], should be modified with dataset
        num_points=(1024, 256, 64, 16, 1), # TODO: (michbaum) We want a single feature vector per mask
        radius=(0.1, 0.2, 0.4, 0.8, 1.6),
        num_samples=(32, 32, 32, 32, 32), # TODO: (michbaum) Probably the last can be 16
        sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256, 512), (512, 512, 1024)),
        fp_channels=(),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max', 
            use_xyz=True,
            normalize_xyz=False)),
    neck=dict(
        type='CUPIDNeck', # TODO: (michbaum) Novel neck that simply builds all 2 pairs between the instances
        # in_channels=1024,
        matching_range=matching_range,
        use_xyz=use_xyz,
        ),
    decode_head=dict(
        # TODO: (michbaum) Needs to change to a simple classifier with 2048 input channels and 2 output channels "match", "no_match"
        type='CUPIDHead',
        num_classes=2,
        balance_classes=balance_classes, # (michbaum) We have vastly more negative "no_match" samples, so we should balance this
        ignore_index=2, # (michbaum) This ignore index is for the binary classification in matching, not the original classes
        feature_size=2048 if not use_xyz else 2054,
        channels=512,
        dropout_ratio=0.5,
        # dropout_ratio=0.0,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(
            type='mmdet.CrossEntropyLoss', # TODO: (michbaum) Maybe change to FocalLoss
            use_sigmoid=False, # (michbaum) Our result is not compatible here even though it's binary classification
            class_weight=(20, 1) if not balance_classes else None, # (michbaum) Either manually weight the classes or balance the samples
            loss_weight=1.0,
            avg_non_ignore=True)),# (michbaum) Make sure to only average the loss over the relevant samples
        # loss_decode=dict(
        #     type='mmdet.FocalLoss', # TODO: (michbaum) Maybe change to FocalLoss
        #     use_sigmoid=True, # (michbaum) Necessary for focal loss
        #     loss_weight=1.0,
        #     )),
    # (michbaum) If given, ignores the other configs and performs heuristic matching inference
    heuristic=dict(
        type='NumNearPoints',
        near_point_threshold=near_point_threshold,
        min_number_near_points=min_number_near_points),
    # model training and testing settings
    train_cfg=dict(),
    # TODO: (michbaum) Adapt the slide approach for more accuracy/adaptability
    test_cfg=dict(
        mode='whole', # TODO: (michbaum) Need 'whole' to use whole scene, otherwise will be 'slide'
        num_points=8192, # (michbaum) This and below is for 'slide' approach
        block_size=1.5,
        sample_rate=0.5,
        use_normalized_coord=False,
        batch_size=24)
        )


# data settings
train_dataloader = dict(batch_size=4, num_workers=4)

# runtime settings
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5))
train_cfg = dict(val_interval=5)