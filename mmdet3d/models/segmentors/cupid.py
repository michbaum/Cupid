# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import Dict, List, Union

from mmengine.model import BaseModel
from mmengine.model import BaseModule

import numpy as np
import torch
from torch import Tensor
from torch import nn as nn

from mmdet3d.structures import PointData
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from ...structures.det3d_data_sample import OptSampleList, SampleList
from ..utils import add_prefix
from .base import Base3DSegmentor


@MODELS.register_module()
class CUPID(Base3DSegmentor):
    """3D Encoder Decoder object instance matcher.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

    loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
    _decode_head_forward_train(): decode_head.loss()
    _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict object matching results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``Det3DDataSample`` including ``pred_pts_seg``.

    .. code:: text

    predict(): inference() -> postprocess_result()
    inference(): whole_inference()/slide_inference()
    whole_inference()/slide_inference(): encoder_decoder()
    encoder_decoder(): extract_feat() -> decode_head.predict()

    4 The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head forward function to forward decode head model.

    .. code:: text

    _forward(): extract_feat() -> _decode_head.forward()

    Args:
        backbone (dict or :obj:`ConfigDict`): The config for the backnone of
            segmentor.
        decode_head (dict or :obj:`ConfigDict`): The config for the decode
            head of segmentor.
        neck (dict or :obj:`ConfigDict`, optional): The config for the neck of
            segmentor. Defaults to None.
        auxiliary_head (dict or :obj:`ConfigDict` or List[dict or
            :obj:`ConfigDict`], optional): The config for the auxiliary head of
            segmentor. Defaults to None.
        loss_regularization (dict or :obj:`ConfigDict` or List[dict or
            :obj:`ConfigDict`], optional): The config for the regularization
            loass. Defaults to None.
        train_cfg (dict or :obj:`ConfigDict`, optional): The config for
            training. Defaults to None.
        test_cfg (dict or :obj:`ConfigDict`, optional): The config for testing.
            Defaults to None.
        data_preprocessor (dict or :obj:`ConfigDict`, optional): The
            pre-process config of :class:`BaseDataPreprocessor`.
            Defaults to None.
        init_cfg (dict or :obj:`ConfigDict` or List[dict or :obj:`ConfigDict`],
            optional): The weight initialized config for :class:`BaseModule`.
            Defaults to None.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 heuristic: OptMultiConfig = None,
                 auxiliary_head: OptMultiConfig = None,
                 loss_regularization: OptMultiConfig = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(CUPID, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self._init_loss_regularization(loss_regularization)

        self.heuristic_inference = False
        if heuristic is not None:
            self.heuristic = MODELS.build(heuristic)
            self.heuristic_inference = True

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head, \
            '3D EncoderDecoder Segmentor should have a decode_head'

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``."""
        self.decode_head = MODELS.build(decode_head)
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self,
                             auxiliary_head: OptMultiConfig = None) -> None:
        """Initialize ``auxiliary_head``."""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def _init_loss_regularization(self,
                                  loss_regularization: OptMultiConfig = None
                                  ) -> None:
        """Initialize ``loss_regularization``."""
        if loss_regularization is not None:
            if isinstance(loss_regularization, list):
                self.loss_regularization = nn.ModuleList()
                for loss_cfg in loss_regularization:
                    self.loss_regularization.append(MODELS.build(loss_cfg))
            else:
                self.loss_regularization = MODELS.build(loss_regularization)

    def extract_feat(self, batch_inputs: Tensor) -> dict:
        """Extract features from points."""
        # (michbaum) Batch_input shape is (B, num_instances, points_per_instance ~ N, 3+C)
        # (michbaum) We want to apply the feature extraction on every instance, so we reshape twice
        flattened_batch = batch_inputs.view(-1, batch_inputs.shape[-2], batch_inputs.shape[-1])
        # (michbaum) Flattened shape (B * num_instances, points_per_instance ~ N, 3+C)
        x = self.backbone(flattened_batch)
        # (michbaum) Reshape back to get feature vector per instance
        for key, value in x.items():
            x[key] = [val.view(batch_inputs.shape[0], batch_inputs.shape[1], -1) for val in value]
        # x = x.view(batch_inputs.shape[0], batch_inputs.shape[1], batch_inputs.shape[-2], batch_inputs.shape[-1]) 
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, batch_inputs: Tensor,
                      batch_input_metas: List[dict]) -> Tensor:
        """Encode points with backbone and decode into a semantic segmentation
        map of the same size as input.

        Args:
            batch_input (Tensor): Input point cloud sample
            batch_input_metas (List[dict]): Meta information of a batch of
                samples.

        Returns:
            Tensor: Segmentation logits of shape [B, num_classes, N].
        """
        x = self.extract_feat(batch_inputs)
        match_logits, feature_indices, distance_bools = \
            self.decode_head.predict(x, batch_input_metas, self.test_cfg)
        return match_logits, feature_indices, distance_bools

    def _decode_head_forward_train(
            self, batch_inputs_dict: dict,
            batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Run forward function and calculate loss for decode head in training.

        Args:
            batch_input (Tensor): Input point cloud sample
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components for decode head.
        """
        losses = dict()
        loss_decode = self.decode_head.loss(batch_inputs_dict,
                                            batch_data_samples, self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(
        self,
        batch_inputs_dict: dict,
        batch_data_samples: SampleList,
    ) -> Dict[str, Tensor]:
        """Run forward function and calculate loss for auxiliary head in
        training.

        Args:
            batch_input (Tensor): Input point cloud sample
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components for auxiliary
            head.
        """
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(batch_inputs_dict, batch_data_samples,
                                         self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(batch_inputs_dict,
                                                batch_data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _loss_regularization_forward_train(self) -> Dict[str, Tensor]:
        """Calculate regularization loss for model weight in training."""
        losses = dict()
        if isinstance(self.loss_regularization, nn.ModuleList):
            for idx, regularize_loss in enumerate(self.loss_regularization):
                loss_regularize = dict(
                    loss_regularize=regularize_loss(self.modules()))
                losses.update(add_prefix(loss_regularize, f'regularize_{idx}'))
        else:
            loss_regularize = dict(
                loss_regularize=self.loss_regularization(self.modules()))
            losses.update(add_prefix(loss_regularize, 'regularize'))

        return losses

    def loss(self, batch_inputs_dict: dict,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): Input sample dict which
                includes 'points' and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        # extract features using backbone
        points = torch.stack(batch_inputs_dict['points'])
        x = self.extract_feat(points)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, batch_data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, batch_data_samples)
            losses.update(loss_aux)

        if self.with_regularization_loss:
            loss_regularize = self._loss_regularization_forward_train()
            losses.update(loss_regularize)

        return losses

    @staticmethod
    def _input_generation(coords,
                          patch_center: Tensor,
                          coord_max: Tensor,
                          feats: Tensor,
                          use_normalized_coord: bool = False) -> Tensor:
        """Generating model input.

        Generate input by subtracting patch center and adding additional
        features. Currently support colors and normalized xyz as features.

        Args:
            coords (Tensor): Sampled 3D point coordinate of shape [S, 3].
            patch_center (Tensor): Center coordinate of the patch.
            coord_max (Tensor): Max coordinate of all 3D points.
            feats (Tensor): Features of sampled points of shape [S, C].
            use_normalized_coord (bool): Whether to use normalized xyz as
                additional features. Defaults to False.

        Returns:
            Tensor: The generated input data of shape [S, 3+C'].
        """
        # subtract patch center, the z dimension is not centered
        centered_coords = coords.clone()
        centered_coords[:, 0] -= patch_center[0]
        centered_coords[:, 1] -= patch_center[1]

        # normalized coordinates as extra features
        if use_normalized_coord:
            normalized_coord = coords / coord_max
            feats = torch.cat([feats, normalized_coord], dim=1)

        points = torch.cat([centered_coords, feats], dim=1)

        return points

    def _sliding_patch_generation(self,
                                  points: Tensor,
                                  num_points: int,
                                  block_size: float,
                                  sample_rate: float = 0.5,
                                  use_normalized_coord: bool = False,
                                  eps: float = 1e-3) -> tuple[Tensor, Tensor]:
        """Sampling points in a sliding window fashion.

        First sample patches to cover all the input points.
        Then sample points in each patch to batch points of a certain number.

        Args:
            points (Tensor): Input points of shape [N, 3+C].
            num_points (int): Number of points to be sampled in each patch.
            block_size (float): Size of a patch to sample.
            sample_rate (float): Stride used in sliding patch. Defaults to 0.5.
            use_normalized_coord (bool): Whether to use normalized xyz as
                additional features. Defaults to False.
            eps (float): A value added to patch boundary to guarantee points
                coverage. Defaults to 1e-3.

        Returns:
            Tuple[Tensor, Tensor]:

            - patch_points (Tensor): Points of different patches of shape
              [K, N, 3+C].
            - patch_idxs (Tensor): Index of each point in `patch_points` of
              shape [K, N].
        """
        device = points.device
        # we assume the first three dims are points' 3D coordinates
        # and the rest dims are their per-point features
        coords = points[:, :3]
        feats = points[:, 3:]

        coord_max = coords.max(0)[0]
        coord_min = coords.min(0)[0]
        stride = block_size * sample_rate
        num_grid_x = int(
            torch.ceil((coord_max[0] - coord_min[0] - block_size) /
                       stride).item() + 1)
        num_grid_y = int(
            torch.ceil((coord_max[1] - coord_min[1] - block_size) /
                       stride).item() + 1)

        patch_points, patch_idxs = [], []
        for idx_y in range(num_grid_y):
            s_y = coord_min[1] + idx_y * stride
            e_y = torch.min(s_y + block_size, coord_max[1])
            s_y = e_y - block_size
            for idx_x in range(num_grid_x):
                s_x = coord_min[0] + idx_x * stride
                e_x = torch.min(s_x + block_size, coord_max[0])
                s_x = e_x - block_size

                # extract points within this patch
                cur_min = torch.tensor([s_x, s_y, coord_min[2]]).to(device)
                cur_max = torch.tensor([e_x, e_y, coord_max[2]]).to(device)
                cur_choice = ((coords >= cur_min - eps) &
                              (coords <= cur_max + eps)).all(dim=1)

                if not cur_choice.any():  # no points in this patch
                    continue

                # sample points in this patch to multiple batches
                cur_center = cur_min + block_size / 2.0
                point_idxs = torch.nonzero(cur_choice, as_tuple=True)[0]
                num_batch = int(np.ceil(point_idxs.shape[0] / num_points))
                point_size = int(num_batch * num_points)
                replace = point_size > 2 * point_idxs.shape[0]
                num_repeat = point_size - point_idxs.shape[0]
                if replace:  # duplicate
                    point_idxs_repeat = point_idxs[torch.randint(
                        0, point_idxs.shape[0],
                        size=(num_repeat, )).to(device)]
                else:
                    point_idxs_repeat = point_idxs[torch.randperm(
                        point_idxs.shape[0])[:num_repeat]]

                choices = torch.cat([point_idxs, point_idxs_repeat], dim=0)
                choices = choices[torch.randperm(choices.shape[0])]

                # construct model input
                point_batches = self._input_generation(
                    coords[choices],
                    cur_center,
                    coord_max,
                    feats[choices],
                    use_normalized_coord=use_normalized_coord)

                patch_points.append(point_batches)
                patch_idxs.append(choices)

        patch_points = torch.cat(patch_points, dim=0)
        patch_idxs = torch.cat(patch_idxs, dim=0)

        # make sure all points are sampled at least once
        assert torch.unique(patch_idxs).shape[0] == points.shape[0], \
            'some points are not sampled in sliding inference'

        return patch_points, patch_idxs

    def slide_inference(self, point: Tensor, input_meta: dict,
                        rescale: bool) -> Tensor:
        """Inference by sliding-window with overlap.

        Args:
            point (Tensor): Input points of shape [N, 3+C].
            input_meta (dict): Meta information of input sample.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.

        Returns:
            Tensor: The output segmentation map of shape [num_classes, N].
        """
        num_points = self.test_cfg.num_points
        block_size = self.test_cfg.block_size
        sample_rate = self.test_cfg.sample_rate
        use_normalized_coord = self.test_cfg.use_normalized_coord
        batch_size = self.test_cfg.batch_size * num_points

        # patch_points is of shape [K*N, 3+C], patch_idxs is of shape [K*N]
        patch_points, patch_idxs = self._sliding_patch_generation(
            point, num_points, block_size, sample_rate, use_normalized_coord)
        feats_dim = patch_points.shape[1]
        seg_logits = []  # save patch predictions

        for batch_idx in range(0, patch_points.shape[0], batch_size):
            batch_points = patch_points[batch_idx:batch_idx + batch_size]
            batch_points = batch_points.view(-1, num_points, feats_dim)
            # batch_seg_logit is of shape [B, num_classes, N]
            batch_seg_logit = self.encode_decode(batch_points,
                                                 [input_meta] * batch_size)
            batch_seg_logit = batch_seg_logit.transpose(1, 2).contiguous()
            seg_logits.append(batch_seg_logit.view(-1, self.num_classes))

        # aggregate per-point logits by indexing sum and dividing count
        seg_logits = torch.cat(seg_logits, dim=0)  # [K*N, num_classes]
        expand_patch_idxs = patch_idxs.unsqueeze(1).repeat(1, self.num_classes)
        preds = point.new_zeros((point.shape[0], self.num_classes)).\
            scatter_add_(dim=0, index=expand_patch_idxs, src=seg_logits)
        count_mat = torch.bincount(patch_idxs)
        preds = preds / count_mat[:, None]

        # TODO: if rescale and voxelization segmentor

        return preds.transpose(0, 1)  # to [num_classes, K*N]

    def whole_inference(self, points: Tensor, batch_input_metas: List[dict],
                        rescale: bool) -> Tensor:
        """Inference with full scene (one forward pass without sliding)."""
        match_logit, feature_indices, distance_bools = self.encode_decode(points, batch_input_metas)
        # TODO: if rescale and voxelization segmentor
        return match_logit, feature_indices, distance_bools

    def inference(self, points: Tensor, batch_input_metas: List[dict],
                  rescale: bool) -> Tensor:
        """Inference with slide/whole style.

        Args:
            points (Tensor): Input points of shape [B, N, 3+C].
            batch_input_metas (List[dict]): Meta information of a batch of
                samples.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.

        Returns:
            Tensor: The output matching map.
        """
        feature_indices = None
        distance_bools = None
        assert self.test_cfg.mode in ['slide', 'whole']
        if self.test_cfg.mode == 'slide':
            # TODO: (michbaum) Needs adaptation to pass the feature indices & distance bools
            #                  correctly
            raise NotImplementedError('Slide inference is not implemented yet.')
            match_logit = torch.stack([
                self.slide_inference(point, input_meta, rescale)
                for point, input_meta in zip(points, batch_input_metas)
            ], 0)
        else:
            match_logit, feature_indices, distance_bools = \
                self.whole_inference(points, batch_input_metas, rescale)
        return match_logit, feature_indices, distance_bools

    def postprocess_result(self, match_logits_list: List[Tensor],
                           feature_indices_list: List[Tensor],
                           distance_bools_list: List[Tensor],
                           batch_data_samples: SampleList) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Args:
            match_logits_list (List[Tensor]): List of matching results,
                match_logits from model of each input instance pointcloud pair.
            feature_indices_list (List[Tensor]): List of feature indices of
                each input instance pointcloud pair.
            distance_bools_list (List[Tensor]): List signifying whether the
                feature vectors of the instance masks are within the matching
                range.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            List[:obj:`Det3DDataSample`]: Matching results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """
        assert len(match_logits_list) == len(feature_indices_list) == len(
            distance_bools_list), "Something in the forward result processing went wrong."
        for i in range(len(match_logits_list)):
            match_logits = match_logits_list[i]
            feature_indices = feature_indices_list[i]
            distance_bools = distance_bools_list[i]
            match_pred = match_logits.argmax(dim=1)
            batch_data_samples[i].set_data({
                'pair_matching_logits':
                PointData(**{'pair_matching_logits': match_logits}),
                'pred_pair_matching':
                PointData(**{'pred_pair_matching': match_pred}),
                'feature_pair_indices':
                PointData(**{'feature_pair_indices': feature_indices}),
                'pair_distance_bools':
                PointData(**{'pair_distance_bools': distance_bools})
            })
        return batch_data_samples

    def _build_pointcloud_pairs(self, pcs):
        """
        Builds all possible pairs of instance pointclouds.

        Args:
            pcs (torch.tensor): Tensor containing the pointclouds with shape [B, num_instances, num_points, point_dim]
        """
        # Get the size parameters         
        
        batch_size, num_pcs, _, _ = pcs.shape

        # Create a meshgrid of indices for the pairs
        i_indices, j_indices = torch.triu_indices(num_pcs, num_pcs, offset=1)

        # Expand the indices to cover all batches
        i_indices = i_indices.unsqueeze(0).expand(batch_size, -1)
        j_indices = j_indices.unsqueeze(0).expand(batch_size, -1)

        # Gather the pairs
        first_vectors = pcs[torch.arange(batch_size)[:, None], i_indices]
        second_vectors = pcs[torch.arange(batch_size)[:, None], j_indices]

        # Concatenate the pairs along a new dimension
        concatenated_pairs = torch.stack([first_vectors, second_vectors], dim=2)

        # (michbaum) To work with the general framework, we need to populate the distance_bools
        #            This just means we consider all combinations and don't exclude any based on
        #            the feature distance - since that's not applicable here anyways
        distance_bools = torch.ones_like(i_indices, dtype=torch.bool)

        # Save the index pairs
        index_pairs = torch.stack((i_indices, j_indices), dim=2)  # shape: (batch_size, num_pairs, 2)

        return concatenated_pairs, index_pairs, distance_bools

    def predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Simple test with single scene.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            rescale (bool): Whether transform to original number of points.
                Will be used for voxelization based segmentors.
                Defaults to True.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pair_matching`` (PointData): Prediction of instance
                matching.
            - ``pair_matching_logits`` (PointData): Predicted logits of instance
                matching before normalization.
            - ``feature_pair_indices`` (PointData): Feature indices of the
                instance pairs.
            - ``pair_distance_bools`` (PointData): Signifying whether the pairs
                are within the matching range.
        """
        # (michbaum) Honestly, don't know if we can batch our prediction
        #            This comment is legacy from the segmentor base
        # 3D matching requires per-pair prediction, so it's impossible
        # to use down-sampling to get a batch of scenes with same num_pairs
        # therefore, we only support testing one scene every time

        match_logits_list = []
        feature_indices_list = []
        distance_bools_list = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)

        points = batch_inputs_dict['points'] 
        if not self.heuristic_inference:
            for point, input_meta in zip(points, batch_input_metas):
                match_logits, feature_indices, distance_bools = self.inference(
                    point.unsqueeze(0), [input_meta], rescale)
                match_logits_list.append(match_logits)
                feature_indices_list.append(feature_indices)
                distance_bools_list.append(distance_bools)
        else:
            # (michbaum) Heuristic inference
            for point, input_meta in zip(points, batch_input_metas):
                concatenated_pointclouds, feature_indices, distance_bools = self._build_pointcloud_pairs(
                    point.unsqueeze(0))
                match_logits = self.heuristic(concatenated_pointclouds)
                match_logits_list.append(match_logits)
                feature_indices_list.append(feature_indices)
                distance_bools_list.append(distance_bools)

        return self.postprocess_result(match_logits_list, 
                                       feature_indices_list,
                                       distance_bools_list,
                                       batch_data_samples)

    def _forward(self,
                 batch_inputs_dict: dict,
                 batch_data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            batch_inputs_dict (dict): Input sample dict which includes 'points'
                and 'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        points = torch.stack(batch_inputs_dict['points'])
        x = self.extract_feat(points)
        return self.decode_head.forward(x)


@MODELS.register_module()
class NumNearPoints(BaseModule):
    r"""Heuristic Pointcloud Matching Module used in CUPID. Counts the number of points between
    two pointclouds that are 'near' each other (in a ball distance). Gives higher scores to pairs
    that share more 'near' points.

    Args:
        near_point_threshold (float): Maximum euclidean distance between points of the
        considered pair of pointclouds to be considered 'near'.
        min_number_near_points (int): Minimum number of points that must be deemed
        'near' between the pair of pointclouds.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self, near_point_threshold, min_number_near_points, init_cfg=None):
        super(NumNearPoints, self).__init__(init_cfg=init_cfg)

        self.near_point_threshold = near_point_threshold
        self.min_number_near_points = min_number_near_points

    def compute_euclidean_distances(self, pc1, pc2):
        """
        Compute the Euclidean distance between each pair of points in two point clouds.
        
        Parameters:
        pc1 (np.ndarray): First point cloud of shape (num_points, 3)
        pc2 (np.ndarray): Second point cloud of shape (num_points, 3)
        
        Returns:
        np.ndarray: Matrix of distances of shape (num_points, num_points)
        """
        diff = pc1[:, np.newaxis, :] - pc2[np.newaxis, :, :]
        dist = torch.sqrt(torch.sum(diff ** 2, axis=-1))
        return dist

    def count_close_points(self, pointclouds):
        """
        Count the number of point pairs within a given distance threshold between two point clouds.
        
        Parameters:
        pointclouds (np.ndarray): Stacked point clouds of shape [B, num_pairs, 2, num_points, point_dim]
        
        Returns:
        np.ndarray: Number of point pairs within threshold for each pair of point clouds
        """
        # Initialize result array
        batch_size, num_pairs, _, num_points, num_features = pointclouds.shape
        result = torch.zeros((batch_size, num_pairs))
        
        # Iterate over each pair
        for batch in range(batch_size):
            for i in range(num_pairs):
                pc1 = pointclouds[batch, i, 0, :, :3]  # First point cloud
                pc2 = pointclouds[batch, i, 1, :, :3]  # Second point cloud
                
                # Compute distances
                distances = self.compute_euclidean_distances(pc1, pc2)
                
                # Count pairs within threshold
                count = torch.sum(distances < self.near_point_threshold)
                count = 0 if count < self.min_number_near_points else count
                result[batch][i] = count
        
        return result

    
    def forward(self, pc_pairs: Tensor) -> Tensor:
        """
        Forward pass. Calculates the number of points between the two pointclouds that 
        are 'near' to each other, i.e. that have an euclidean distance less than
        self.near_point_threshold. Returns a score that is higher for pairs that share more
        'near' points. Gives a score of 0 if the number of 'near' points is less than
        self.min_number_near_points.

        Args:
            pc_pairs (Tensor): Paired up pointclouds with shape [B, num_pairs, 2, num_points, point_dim]

        Returns:
            Tensor: Score for each pointcloud pair. Shape [B, num_pairs]
        """
        num_near_points = self.count_close_points(pc_pairs)

        # (michbaum) Transform into our prediction format. Label 0 is match, label 1 is no match
        #            Initialize all fields with 1 and then overwrite the 0 row with the num_points.
        #            Argmax will do the rest.
        score = torch.ones((num_near_points.shape[0], 2, num_near_points.shape[1]))
        score[:, 0, :] = num_near_points

        return score
