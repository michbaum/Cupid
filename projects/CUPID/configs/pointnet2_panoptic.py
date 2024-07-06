# ------------------------PARAMETERS---------------------
clustering_method = 'kmeans'  # (michbaum) 'kmeans' or 'meanshift', meanshift is slower
kmeans_clusters = 20
meanshift_bandwidth = 0.3  # (michbaum) From https://github.com/prs-eth/PanopticSegForMobileMappingPointClouds/blob/main/conf/models/panoptic/pointnet2.yaml#L168 
segmentation_dropout = 0.5
instance_dropout = 0.
embed_loss_weight = 0.5  # (michbaum) 0.1 originally from paper: https://arxiv.org/pdf/2304.13980
instance_overlap_threshold = 0.5  # (michbaum) Score threshold for our matching approach
# ------------------------PARAMETERS---------------------

# -------------------------MODEL-------------------------
model = dict(
    type='CUPIDPanoptic',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    instance_overlap_threshold=instance_overlap_threshold,
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