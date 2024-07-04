# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple

from mmcv.cnn.bricks import ConvModule
import torch
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import PointFPModule
from mmengine.model import normal_init
from mmdet3d.registry import MODELS
from mmdet3d.utils.typing_utils import ConfigType
from mmdet3d.structures.det3d_data_sample import SampleList
from .decode_head import Base3DDecodeHead


@MODELS.register_module()
class PointNet2PanopticHead(Base3DDecodeHead):
    r"""PointNet2 panoptic decoder head.

    Decoder head used in `A Review of Panoptic Segmentation for Mobile Mapping Point Clouds <https://arxiv.org/abs/2304.13980>`_.
    Refer to the `official code <https://github.com/prs-eth/PanopticSegForMobileMappingPointClouds/blob/main/torch_points3d/models/panoptic/pointnet2.py#L10>`_.

    Args:
        fp_channels (Sequence[Sequence[int]]): Tuple of mlp channels in FP
            modules. Defaults to ((768, 256, 256), (384, 256, 256),
            (320, 256, 128), (128, 128, 128, 128)).
        embed_dim (int): The dimension of the embedding feature. Defaults to 5.
        embed_dropout_ratio (float): The ratio of dropout layer in embedding
            layer. Defaults to 0.5.
        fp_norm_cfg (dict or :obj:`ConfigDict`): Config of norm layers used
            in FP modules. Defaults to dict(type='BN2d').
    """

    def __init__(self,
                 fp_channels: Sequence[Sequence[int]] = ((768, 256, 256),
                                                         (384, 256, 256),
                                                         (320, 256, 128),
                                                         (128, 128, 128, 128)),
                 embed_dim: int = 5,
                 clustering_method: str = 'kmeans',
                 kmeans_clusters: int = 20,
                 meanshift_bandwidth: float = 0.6,
                 embed_dropout_ratio: float = 0.5,
                 ins_loss_decode: ConfigType = dict(
                        type='mmdet.DiscriminativeLoss',
                        loss_weight=1.0),
                 fp_norm_cfg: ConfigType = dict(type='BN2d'),
                 **kwargs) -> None:
        super(PointNet2PanopticHead, self).__init__(**kwargs)

        assert clustering_method in ['kmeans', 'meanshift'], "Clustering method must be either 'kmeans' or 'meanshift'!"
        self.clustering_method = clustering_method
        self.kmeans_clusters = kmeans_clusters
        self.embed_dim = embed_dim
        self.meanshift_bandwidth = meanshift_bandwidth
        self.embed_dropout_ratio = embed_dropout_ratio
        self.ins_loss_decode = MODELS.build(ins_loss_decode)
        self.num_fp = len(fp_channels)
        self.FP_modules = nn.ModuleList()
        for cur_fp_mlps in fp_channels:
            self.FP_modules.append(
                PointFPModule(mlp_channels=cur_fp_mlps, norm_cfg=fp_norm_cfg))

        # https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_sem_seg.py#L40
        self.pre_seg_conv = ConvModule(
            fp_channels[-1][-1],
            self.channels,
            kernel_size=1,
            bias=True,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        # https://github.com/prs-eth/PanopticSegForMobileMappingPointClouds/blob/main/torch_points3d/models/panoptic/pointnet2.py#L87
        self.pre_embed_conv = ConvModule(
            fp_channels[-1][-1],
            self.channels,
            kernel_size=1,
            bias=True,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # TODO: (michbaum) Maybe need to init weight
        self.conv_ins_seg = self.build_conv_seg(
            channels=self.channels,
            num_classes=self.embed_dim,
            kernel_size=1)
        
        if self.embed_dropout_ratio > 0:
            self.embed_dropout = nn.Dropout(self.embed_dropout_ratio)
        else:
            self.embed_dropout = None
        

    def _extract_input(self,
                       feat_dict: dict) -> Tuple[List[Tensor], List[Tensor]]:
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            Tuple[List[Tensor], List[Tensor]]: Coordinates and features of
            multiple levels of points.
        """
        sa_xyz = feat_dict['sa_xyz']
        sa_features = feat_dict['sa_features']
        assert len(sa_xyz) == len(sa_features)

        return sa_xyz, sa_features

    def init_weights(self) -> None:
        """Initialize weights of classification layer."""
        super().init_weights()
        normal_init(self.conv_ins_seg, mean=0, std=0.01)

    def ins_seg(self, feat: Tensor) -> Tensor:
        """Embed each point."""
        if self.embed_dropout is not None:
            feat = self.embed_dropout(feat)
        output = self.conv_ins_seg(feat)
        return output

    def forward(self, feat_dict: dict) -> Tensor:
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            Tensor: Segmentation map of shape [B, num_classes, N].
        """
        sa_xyz, sa_features = self._extract_input(feat_dict)

        # https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_sem_seg.py#L24
        sa_features[0] = None

        fp_feature = sa_features[-1]

        for i in range(self.num_fp):
            # consume the points in a bottom-up manner
            fp_feature = self.FP_modules[i](sa_xyz[-(i + 2)], sa_xyz[-(i + 1)],
                                            sa_features[-(i + 2)], fp_feature)

        output_sem = self.cls_seg(self.pre_seg_conv(fp_feature))
        output_ins = self.ins_seg(self.pre_embed_conv(fp_feature))

        output = {'sem_logit': output_sem,
                  'ins_logit': output_ins}

        return output

    # TODO: (michbaum) Change
    def loss(self, inputs: dict, batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict[str, Tensor]:
        """Forward function for training.

        Args:
            inputs (dict): Feature dict from backbone.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            train_cfg (dict or :obj:`ConfigDict`): The training config.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_pts_seg.pts_semantic_mask
            for data_sample in batch_data_samples
        ]
        gt_instance_segs = [
            data_sample.gt_pts_seg.pts_instance_mask
            for data_sample in batch_data_samples
        ]

        return torch.stack(gt_semantic_segs, dim=0), torch.stack(gt_instance_segs, dim=0)

    def loss_by_feat(self, seg_logit: dict[Tensor],
                     batch_data_samples: SampleList) -> dict[str, Tensor]:
        """Compute semantic segmentation loss.

        Args:
            seg_logit (dict[Tensor]): Predicted per-point semantic 
                segmentation logits and instance segmentation embedding
                of shape [B, num_classes, N] and [B, embed_dim, N], respectively.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        sem_seg_label, ins_seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        loss['loss_sem_seg'] = self.loss_decode(
            seg_logit['sem_logit'], sem_seg_label, ignore_index=self.ignore_index)
        # TODO: (michbaum) Need to make sure to keep instances of different classes separate!
        loss.update(**self.ins_loss_decode(
            seg_logit['ins_logit'], sem_seg_label, ins_seg_label, self.embed_dim, ignore_idx=self.ignore_index))
        return loss
