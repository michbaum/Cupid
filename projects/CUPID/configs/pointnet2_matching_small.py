base_ = [
    './pointnet2_matching.py',
]

model = dict(
    backbone=dict(
        num_points=(512, 64, 8, 1), # TODO: (michbaum) We want a single feature vector per mask
        radius=(0.1, 0.2, 0.4, 0.8),
        num_samples=(32, 32, 32, 32), # TODO: (michbaum) Probably the last can be 16
        sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256, 512)),
        sa_cfg=dict(
            pool_mod='max', # TODO: (michbaum) max or avg
            use_xyz=True,
            normalize_xyz=False)
    ),
    decode_head=dict(
        feature_size=1024,
        # loss_decode=dict(
        #     _delete_=True,
        #     type='mmdet.FocalLoss', # TODO: (michbaum) Maybe change to FocalLoss
        #     use_sigmoid=True, # (michbaum) Necessary for focal loss
        #     loss_weight=1.0, # (michbaum) Focal loss downweighs heavily
        #     ),
    ),
)


# data settings
train_dataloader = dict(batch_size=8, num_workers=4)

# runtime settings
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10))
train_cfg = dict(val_interval=10)