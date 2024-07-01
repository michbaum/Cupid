# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import List, Sequence, Tuple, Dict

import torch
from torch import Tensor
from torch import nn as nn

from mmcv.cnn.bricks import ConvModule
from mmengine.model import BaseModule, normal_init
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils.typing_utils import ConfigType, OptMultiConfig
from mmdet3d.registry import MODELS
from mmdet3d.utils.typing_utils import ConfigType

from mmdet.models.losses import FocalLoss


@MODELS.register_module()
class CUPIDHead(BaseModule, metaclass=ABCMeta):
    r"""CUPID decoder head.

    1. The ``init_weights`` method is used to initialize decode_head's
    model parameters. After segmentor initialization, ``init_weights``
    is triggered when ``segmentor.init_weights()`` is called externally.

    2. The ``loss`` method is used to calculate the loss of decode_head,
    which includes two steps: (1) the decode_head model performs forward
    propagation to obtain the feature maps (2) The ``loss_by_feat`` method
    is called based on the feature maps to calculate the loss.

    .. code:: text

    loss(): forward() -> loss_by_feat()

    3. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) the decode_head model performs forward
    propagation to obtain the feature maps (2) The ``predict_by_feat`` method
    is called based on the feature maps to predict segmentation results
    including post-processing.

    .. code:: text

    predict(): forward() -> predict_by_feat()

    Args:
        feature_size (int): Size of the input feature vectors.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Defaults to 0.5.
        conv_cfg (dict or :obj:`ConfigDict`): Config of conv layers.
            Defaults to dict(type='Conv1d').
        norm_cfg (dict or :obj:`ConfigDict`): Config of norm layers.
            Defaults to dict(type='BN1d').
        act_cfg (dict or :obj:`ConfigDict`): Config of activation layers.
            Defaults to dict(type='ReLU').
        loss_decode (dict or :obj:`ConfigDict`): Config of decode loss.
            Defaults to dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            class_weight=None, loss_weight=1.0).
        conv_seg_kernel_size (int): The kernel size used in conv_seg.
            Defaults to 1.
        ignore_index (int): The label index to be ignored. When using masked
            BCE loss, ignore_index should be set to None. Defaults to 255.
        init_cfg (dict or :obj:`ConfigDict` or list[dict or :obj:`ConfigDict`],
            optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 feature_size: int = 2048,
                 channels: int = 512,
                 num_classes: int = 2,
                 balance_classes: bool = False,
                 dropout_ratio: float = 0.5,
                 conv_cfg: ConfigType = dict(type='Conv1d'),
                 norm_cfg: ConfigType = dict(type='BN1d'),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 loss_decode: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 conv_seg_kernel_size: int = 1,
                 ignore_index: int = 255,
                 init_cfg: OptMultiConfig = None) -> None:

        super(CUPIDHead, self).__init__(init_cfg=init_cfg)
        self.channels = channels
        self.num_classes = num_classes
        self.balance_classes = balance_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.loss_decode = MODELS.build(loss_decode)
        self.ignore_index = ignore_index

        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)
        else:
            self.dropout = None

        self.pre_seg_conv = ConvModule(
            feature_size,
            self.channels,
            kernel_size=1,
            bias=True,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv_seg = self.build_conv_seg(
            channels=channels,
            num_classes=num_classes,
            kernel_size=conv_seg_kernel_size)

    def init_weights(self) -> None:
        """Initialize weights of classification layer."""
        super().init_weights()
        normal_init(self.conv_seg, mean=0, std=0.01)

    def build_conv_seg(self, channels: int, num_classes: int,
                       kernel_size: int) -> nn.Module:
        """Build Convolutional Segmentation Layers."""
        return nn.Conv1d(channels, num_classes, kernel_size=kernel_size)

    def cls_seg(self, feat: Tensor) -> Tensor:
        """Classify each points."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def loss(self, inputs: dict, batch_data_samples: SampleList,
             train_cfg: ConfigType) -> Dict[str, Tensor]:
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
        matching_logits, feature_indices, distance_bools = self.forward(inputs)
        losses = self.loss_by_feat(matching_logits, feature_indices, distance_bools, batch_data_samples)
        return losses

    def predict(self, inputs: dict, batch_input_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for testing.

        Args:
            inputs (dict): Feature dict from backbone.
            batch_input_metas (List[dict]): Meta information of a batch of
                samples.
            test_cfg (dict or :obj:`ConfigDict`): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        matching_logits, feature_indices, distance_bools = self.forward(inputs)

        return matching_logits, feature_indices, distance_bools

    def _extract_input(self,
                       feat_dict: dict) -> Tuple[List[Tensor], List[Tensor]]:
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            Tuple[List[Tensor], List[Tensor]]: Coordinates and features of
            multiple levels of points.
        """
        stacked_features = feat_dict['stacked_features']
        feature_indices = feat_dict['feature_indices']
        distance_bools = feat_dict['distance_bools']
        assert stacked_features.shape[:2] == feature_indices.shape[:2] == distance_bools.shape[:2], "Shapes of stacked_features, feature_indices, and distance_bools must match."

        return stacked_features, feature_indices, distance_bools

    def forward(self, feat_dict: dict) -> Tensor:
        """Forward pass.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            Tensor: Segmentation map of shape [B, num_classes, N].
        """
        stacked_features, feature_indices, distance_bools = self._extract_input(feat_dict)

        # (michbaum) Prepare features (B, C, N) for the classification
        stacked_features = stacked_features.transpose(1, 2)
        # TODO: (michbaum) Might need to transpose the other two as well

        output = self.pre_seg_conv(stacked_features)
        output = self.cls_seg(output)
        # (michbaum) Output shape: (B, num_classes, N)

        return output, feature_indices, distance_bools
    
    def _extract_match_gt(self, batch_data_samples: SampleList, matching_indices: Tensor, distance_bools: Tensor) -> Tensor:
        """
        Build the ground truth matching mask from the metadate and the built pairs.

        Args:
            batch_data_samples (SampleList): Metadata of the samples.
            matching_indices (Tensor): Indices of the matching pairs.
            distance_bools (Tensor): Whether the pairs are within the matching range.

        Returns:
            Tensor: Mask per batch that indicates matches as 0, non-matches as 1 and ignored pairs as ignore_idx.
        """
        match_gt = []

        for idx, batch_data in enumerate(batch_data_samples):
            instance_gt_mapping_i = batch_data.gt_pts_seg.instance_gt_mapping
            pcd_to_instance_mapping_i = batch_data.gt_pts_seg.pcd_to_instance_mapping

            # (michbaum) Build the matching labels: 0 for match, 1 for non-match, ignore_index for pairs outside the matching range
            #            and for non-existent instance masks (that we filled up to a certain max size with -1 originally)
            match_gt_i = torch.ones_like(distance_bools[0], dtype=torch.long) * self.ignore_index
            for i, (pcd1, pcd2) in enumerate(matching_indices[idx]):
                pcd1 = pcd1.item()
                pcd2 = pcd2.item()
                # (michbaum) Check if one of the pointclouds in the pair was a bogus fillup or if the
                #            pair was too far apart in euclidean space
                if pcd1 not in pcd_to_instance_mapping_i or pcd2 not in pcd_to_instance_mapping_i or not distance_bools[idx][i]:
                    match_gt_i[i] = self.ignore_index
                else:
                    instance1 = pcd_to_instance_mapping_i[pcd1]
                    instance2 = pcd_to_instance_mapping_i[pcd2]
                    gt_instance1 = instance_gt_mapping_i[instance1]
                    gt_instance2 = instance_gt_mapping_i[instance2]
                    if gt_instance1 == gt_instance2:
                        match_gt_i[i] = 0
                    else:
                        match_gt_i[i] = 1
            match_gt.append(match_gt_i)    

        return torch.stack(match_gt, dim=0)

    def loss_by_feat(self, match_logit: Tensor, matching_indices: Tensor, distance_bools: Tensor,
                        batch_data_samples: SampleList) -> dict[str, Tensor]:
        """Compute semantic segmentation loss.

        Args:
            match_logit (Tensor): Predicted per-instance-pair matching logits of
                shape [B, num_classes, N].

            matching_indices (Tensor): Indices of the matching pairs.

            distance_bools (Tensor): Whether the pairs are within the matching range.
            
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """
        match_label = self._extract_match_gt(batch_data_samples, matching_indices, distance_bools)
        match_label = match_label.to(match_logit.device)

        # (michbaum) If we want to balance the examples, we need to do it here
        if self.balance_classes:
            # (michbaum) Need to treat each batch individually
            for i in range(match_label.shape[0]):
                match_label_i = match_label[i]
                # (michbaum) Count the number of positive and negative examples
                pos = torch.sum(match_label_i == 0)
                neg = torch.sum(match_label_i == 1)
                # (michbaum) Sample as many negative examples as positive examples, setting the rest to ignore_index
                if pos < neg:
                    neg_indices = torch.where(match_label_i == 1)[0]
                    # (michbaum) Randomly permute the indices
                    rand_perm = torch.randperm(neg_indices.size(0))
                    # (michbaum) Only keep the first pos indices
                    neg_indices = neg_indices[rand_perm][pos:]
                    match_label_i[neg_indices] = self.ignore_index
                elif neg < pos:
                    pos_indices = torch.where(match_label_i == 0)[0]
                    rand_perm = torch.randperm(pos_indices.size(0))
                    pos_indices = pos_indices[rand_perm][neg:]
                    match_label_i[pos_indices] = self.ignore_index

        loss = dict()
        if self.loss_decode.__class__ is not FocalLoss().__class__:
            loss['loss_matching'] = self.loss_decode(
                match_logit, match_label, ignore_index=self.ignore_index)
        else:
            # TODO: (michbaum) To use focal loss, we already need to filter the match_logits & match_labels here
            #                  because there is no ignore_index in focal loss
            valid_mask = match_label != self.ignore_index
            # TODO: (michbaum) Also, focal loss is per batch, so we need to accumulate here
            losses = []
            for i in range(match_logit.shape[0]):
                match_logit_i = match_logit[i].transpose(0,1)
                match_label_i = match_label[i]
                valid_mask_i = valid_mask[i]
                match_logit_i = match_logit_i[valid_mask_i]
                match_label_i = match_label_i[valid_mask_i]
                losses.append(self.loss_decode(
                    match_logit_i, match_label_i))
            loss['loss_matching'] = torch.sum(torch.stack(losses))
        
        return loss
