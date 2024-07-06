# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import Dict, List, Union

from mmengine.model import BaseModel
from mmengine.model import BaseModule

import numpy as np
import torch
from torch import Tensor
from torch import nn as nn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MeanShift, KMeans
import time

from mmdet3d.structures import PointData
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from ...structures.det3d_data_sample import OptSampleList, SampleList
from ..utils import add_prefix
from .base import Base3DSegmentor


@MODELS.register_module()
class CUPIDMatching(Base3DSegmentor):
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
        backbone_pretrained_config (str): The config for the pretrained backbone
            model. Defaults to None.
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
                 backbone_pretrained_config: str = None,
                 freeze_pretrained_backbone: bool = False,
                 neck: OptConfigType = None,
                 heuristic: OptMultiConfig = None,
                 auxiliary_head: OptMultiConfig = None,
                 loss_regularization: OptMultiConfig = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 postprocess_matches: bool = False,
                 postprocess_strategy: str = 'greedy',
                 visualize_fails: bool = False,
                 init_cfg: OptMultiConfig = None) -> None:
        super(CUPIDMatching, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        # (michbaum) Postprocessing
        self.postprocess_matches = postprocess_matches
        self.postprocess_strategy = postprocess_strategy
        self.visualize_fails = visualize_fails

        self.backbone = MODELS.build(backbone)

        # (michbaum) Loading pretraining
        self.freeze_pretrained_backbone = freeze_pretrained_backbone
        if backbone_pretrained_config is not None:
            self._init_backbone_from_pretrained(backbone_pretrained_config)

        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self._init_loss_regularization(loss_regularization)

        # (michbaum) Heuristic inference
        self.heuristic_inference = False
        if heuristic is not None:
            self.heuristic = MODELS.build(heuristic)
            self.heuristic_inference = True

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head, \
            '3D EncoderDecoder Segmentor should have a decode_head'

    def _init_backbone_from_pretrained(self, backbone_pretrained_config: str) -> None:
        """Initialize backbone from pretrained config."""
        # (michbaum) Load the pretrained model and copy weights over
        #            as needed
        pretrained_state_dict = torch.load(backbone_pretrained_config)['state_dict']
        new_model_state_dict = self.backbone.state_dict()

        # Remove prefix 'backbone.' from the pretrained state dict keys
        stripped_pretrained_state_dict = {key.replace('backbone.', ''): value for key, value in pretrained_state_dict.items()}

        for name, param in stripped_pretrained_state_dict.items():
            if name in new_model_state_dict:
                try:
                    new_model_state_dict[name].copy_(param)
                    print(f"Copied {name} from pretrained model to new model.")
                except Exception as e:
                    print(f"Could not copy {name} due to {e}.")
            else:
                print(f"{name} not found in the new model's state dictionary.")

        # Load the modified state dictionary back into the new model
        self.backbone.load_state_dict(new_model_state_dict)

        # Freeze the parameters by setting requires_grad to False
        if self.freeze_pretrained_backbone:
            print("Freezing the pretrained backbone.")
            for name, param in self.backbone.named_parameters():
                if name in stripped_pretrained_state_dict:
                    param.requires_grad = False

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
    
    def greedy_postprocess(self, match_logits_list: List[Tensor],
                           feature_indices_list: List[Tensor],
                           batch_data_samples: SampleList) -> None:
        """Greedy postprocessing of the matching results.

        Args:
            match_logits_list (List[Tensor]): List of matching results,
                match_logits from model of each input instance pointcloud pair.
            feature_indices_list (List[Tensor]): List of feature indices of
                each input instance pointcloud pair.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
        """
        # (michbaum) Iterate over all batches, shape: [1, num_classes=2, num_pairs]
        for i in range(len(match_logits_list)):
            match_logits = match_logits_list[i]
            feature_indices = feature_indices_list[i]
            data_sample = batch_data_samples[i]
            # (michbaum) Greedy postprocessing: 
            #            We greedily choose the match with the highest logit, setting all other
            #            matches with these instances to a value less than the label 1 logit
            #            We do this until all instances with matches have been assigned
            #            or set to no match if their pair was already chosen
            
            match_logits = match_logits.squeeze(0)
            feature_indices = feature_indices.squeeze(0)

            match_probabilities = match_logits[0]
            no_match_probabilities = match_logits[1]
            
            # Initialize a tensor to store the modified logits
            modified_logits = match_logits.clone()

            # Set probabilities where no match logit is greater than match logit to a large negative number
            match_probabilities[match_probabilities <= no_match_probabilities] = float('-inf')

            # Initialize a set to keep track of matched indices
            matched_indices = set()

            # Iterate to greedily select the highest probability matches
            while True:
                # Find the pair with the highest match probability
                max_prob, max_idx = torch.max(match_probabilities, dim=0)

                # Break the loop if all remaining probabilities are negative infinity
                if max_prob == float('-inf'):
                    break

                # Get the indices of the selected pair
                idx1, idx2 = feature_indices[max_idx]

                # Set the selected match probability to negative infinity to avoid selecting it again
                match_probabilities[max_idx] = float('-inf')

                # Need to make sure that the match does not contain a bogus index that was produces by the padding
                pcd_to_instance_mapping = data_sample.eval_ann_info['pcd_to_instance_mapping']
                if idx1.item() not in pcd_to_instance_mapping.keys() or idx2.item() not in pcd_to_instance_mapping.keys():
                    # Remember that they get ignored in the eval anyways (we still turn it into a no match if this ever changes)
                    modified_logits[0, max_idx] = modified_logits[1, max_idx] - 1  # Ensuring match logit is less than no match logit
                    continue

                # Add the indices to the matched set
                matched_indices.add(idx1.item())
                matched_indices.add(idx2.item())


                # Set all other match logits involving these indices to a value smaller than their corresponding no match logit
                for j in range(match_logits.shape[1]):
                    if j == max_idx:
                        # Keep the selected match logit
                        continue
                    if idx1 in feature_indices[j] or idx2 in feature_indices[j]:
                        modified_logits[0, j] = modified_logits[1, j] - 1  # Ensuring match logit is less than no match logit
                        match_probabilities[j] = float('-inf')

            match_logits_list[i] = modified_logits.unsqueeze(0)

    def hungarian_postprocess(self, match_logits_list: List[torch.Tensor],
                            feature_indices_list: List[torch.Tensor],
                            batch_data_samples: List) -> None:
        """Hungarian postprocessing of the matching results.

        Args:
            match_logits_list (List[torch.Tensor]): List of matching results,
                match_logits from model of each input instance pointcloud pair.
            feature_indices_list (List[torch.Tensor]): List of feature indices of
                each input instance pointcloud pair.
            batch_data_samples (List): The det3d data samples. It usually includes
                information such as `metainfo` and `gt_pts_seg`.
        """
        for i in range(len(match_logits_list)):
            match_logits = match_logits_list[i]
            feature_indices = feature_indices_list[i]
            data_sample = batch_data_samples[i]

            match_logits = match_logits.squeeze(0)
            feature_indices = feature_indices.squeeze(0)

            match_probabilities = match_logits[0]
            no_match_probabilities = match_logits[1]

            # Set probabilities where no match logit is greater than match logit to a large negative value
            match_probabilities[match_probabilities <= no_match_probabilities] = -10000.0

            # Get the unique indices in feature_indices
            unique_indices = torch.unique(feature_indices)
            num_unique_indices = len(unique_indices)

            # Initialize the cost matrix for the Hungarian algorithm
            cost_matrix = 10000.0 * torch.ones((num_unique_indices, num_unique_indices))

            # Fill the cost matrix with match probabilities
            for j in range(match_probabilities.shape[0]):
                idx1, idx2 = feature_indices[j]
                if idx1.item() in unique_indices and idx2.item() in unique_indices:
                    row = (unique_indices == idx1).nonzero(as_tuple=True)[0].item()
                    col = (unique_indices == idx2).nonzero(as_tuple=True)[0].item()
                    cost_matrix[row, col] = -match_probabilities[j].item()

            # Convert cost_matrix to numpy for scipy
            cost_matrix_np = cost_matrix.cpu().numpy()

            # Apply the Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix_np)

            # Initialize a tensor to store the modified logits
            modified_logits = match_logits.clone()

            # Set all logits to no match
            modified_logits[0] = no_match_probabilities - 1

            # Assign the selected matches
            for row, col in zip(row_ind, col_ind):
                if cost_matrix_np[row, col] < 10000.0:  # Ensure the match is valid
                    idx1 = unique_indices[row].item()
                    idx2 = unique_indices[col].item()
                    original_pair_idx = ((feature_indices[:, 0] == idx1) & (feature_indices[:, 1] == idx2)).nonzero(as_tuple=True)[0]
                    if len(original_pair_idx) > 0:
                        original_pair_idx = original_pair_idx.item()
                        modified_logits[0, original_pair_idx] = match_logits[0, original_pair_idx]

            match_logits_list[i] = modified_logits.unsqueeze(0)

    def visualize_failures(self, gt_matching_mask, pred_pair_matching_mask, pointcloud_pairs):
        """
        Visualize false negative matches (missed matches) and false positive matches (wrong matches).

        Args:
            gt_matching_masks (list[Tensor]): Ground truth matching masks.
            pred_pair_matching_masks (list[Tensor]): Predicted matching mask.
            pointcloud_pairs_list (list[Tensor]): Associated pointcloud pairs.
        """
        from mmdet3d.visualization import Det3DLocalVisualizer

        # We get the indices of the false negatives (gt 0 but pred 1)
        false_negatives = (gt_matching_mask == 0) & (pred_pair_matching_mask.numpy() == 1)
        false_neg_indices = torch.nonzero(torch.from_numpy(false_negatives[0]))
        
        # We get the indices of the false positives (gt 1 but pred 0)
        false_positives = (gt_matching_mask == 1) & (pred_pair_matching_mask.numpy() == 0)
        false_pos_indices = torch.nonzero(torch.from_numpy(false_positives[0]))

        # We visualize the associated pointcloud pairs
        for i in false_neg_indices:
            print("Looking at false negatives")
            pc_1, pc_2 = pointcloud_pairs[0][i.item()]
            # (michbaum) Combine the pointclouds
            pc = torch.cat([pc_1, pc_2], dim=0)
            visualizer = Det3DLocalVisualizer()
            visualizer.set_points(np.asarray(pc.cpu()), pcd_mode=2, vis_mode='add', mode='xyzrgb')
            visualizer.show()
            visualizer._clear_o3d_vis()

        # # We visualize the associated pointcloud pairs
        for j in false_pos_indices:
            print("Looking at false positives")
            pc_1, pc_2 = pointcloud_pairs[0][j.item()]
            # (michbaum) Combine the pointclouds
            pc = torch.cat([pc_1, pc_2], dim=0)
            visualizer = Det3DLocalVisualizer()
            visualizer.set_points(np.asarray(pc.cpu()), pcd_mode=2, vis_mode='add', mode='xyzrgb')
            visualizer.show()
            visualizer._clear_o3d_vis()

    def _extract_match_gt(self, instance_gt_mapping: Tensor, pcd_to_instance_mapping: Tensor, 
                          matching_indices: Tensor, distance_bools: Tensor, ignore_index: int) -> Tensor:
        """
        Build the ground truth matching mask from the metadata and the built pairs.

        Args:
            instance_gt_mapping (Tensor): Mapping from instance to ground truth class.
            pcd_to_instance_mapping (Tensor): Mapping from pointcloud to instance.
            matching_indices (Tensor): Indices of the matching pairs.
            distance_bools (Tensor): Whether the pairs are within the matching range.
            ignore_index (int): Index to ignore in the mask.

        Returns:
            Tensor: Mask per batch that indicates matches as 0, non-matches as 1 and ignored pairs as ignore_idx.
        """
        # (michbaum) Build the matching labels: 0 for match, 1 for non-match, ignore_index for pairs outside the matching range
        #            and for non-existent instance masks (that we filled up to a certain max size with -1 originally)
        match_gt = torch.ones_like(distance_bools[0], dtype=torch.long) * ignore_index
        for i, (pcd1, pcd2) in enumerate(matching_indices[0]):
            pcd1 = pcd1.item()
            pcd2 = pcd2.item()
            # (michbaum) Check if one of the pointclouds in the pair was a bogus fillup or if the
            #            pair was too far apart in euclidean space
            if pcd1 not in pcd_to_instance_mapping or pcd2 not in pcd_to_instance_mapping or not distance_bools[0][i]:
                match_gt[i] = ignore_index
            else:
                instance1 = pcd_to_instance_mapping[pcd1]
                instance2 = pcd_to_instance_mapping[pcd2]
                gt_instance1 = instance_gt_mapping[instance1]
                gt_instance2 = instance_gt_mapping[instance2]
                if gt_instance1 == gt_instance2:
                    match_gt[i] = 0
                else:
                    match_gt[i] = 1

        return match_gt


    def postprocess_result(self, match_logits_list: List[Tensor],
                           feature_indices_list: List[Tensor],
                           distance_bools_list: List[Tensor],
                           batch_data_samples: SampleList,
                           pointclouds = []) -> SampleList:
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
        
        if self.postprocess_matches:
            # (michbaum) Postprocess the results if desired. This makes sure that only
            #            1 match per instance is predicted, either greedily or with
            #            the Hungarian algorithm predicting the 'optimal' matches
            assert self.postprocess_strategy in ['greedy', 'hungarian'], \
                "Invalid postprocessing strategy. Choose either 'greedy' or 'hungarian'."
            
            if self.postprocess_strategy == 'greedy':
                self.greedy_postprocess(match_logits_list, feature_indices_list, batch_data_samples)
            elif self.postprocess_strategy == 'hungarian':
                self.hungarian_postprocess(match_logits_list, feature_indices_list, batch_data_samples)

        for i in range(len(match_logits_list)):
            match_logits = match_logits_list[i]
            feature_indices = feature_indices_list[i]
            distance_bools = distance_bools_list[i]
            match_pred = match_logits.argmax(dim=1)


            # (michbaum) Visualization shit
            if pointclouds != []:
                # (michbaum) Needed for viz
                pointcloud = pointclouds[i]

                instance_gt_mapping = batch_data_samples[i].eval_ann_info['instance_gt_mapping']
                pcd_to_instance_mapping = batch_data_samples[i].eval_ann_info['pcd_to_instance_mapping']
            
                gt_matching_mask = np.asarray(self._extract_match_gt(instance_gt_mapping,
                                                            pcd_to_instance_mapping,
                                                            feature_indices,
                                                            distance_bools,
                                                            2))

                # (michbaum) Visualization of wrong matches
                self.visualize_failures(gt_matching_mask, match_pred, pointcloud) 

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

        # (michbaum) Check that the stacking worked correctly - True
        # check1 = [concatenated_pairs[0][idx][0] == pcs[0][i.item()] for idx, (i, j) in enumerate(index_pairs[0])]
        # check2 = [concatenated_pairs[0][idx][1] == pcs[0][j.item()] for idx, (i, j) in enumerate(index_pairs[0])]
        # assert all([torch.all(a) for a in check1]) and all([torch.all(b) for b in check2]), "Stacking of pointcloud pairs went wrong."

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
        
        # (michbaum) Needed for viz
        pointclouds = []

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
                
                # (michbaum) Need for viz
                if self.visualize_fails:
                    pointclouds.append(concatenated_pointclouds)

                match_logits = self.heuristic(concatenated_pointclouds)
                match_logits_list.append(match_logits)
                feature_indices_list.append(feature_indices) # (michbaum) They are wrong here
                distance_bools_list.append(distance_bools)

        return self.postprocess_result(match_logits_list, 
                                       feature_indices_list,
                                       distance_bools_list,
                                       batch_data_samples,
                                       pointclouds)

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


@MODELS.register_module()
class CUPIDPanoptic(Base3DSegmentor):
    """3D Encoder Decoder panoptic segmentor.

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

    2. The ``predict`` method is used to predict segmentation results,
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
                 auxiliary_head: OptMultiConfig = None,
                 loss_regularization: OptMultiConfig = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(CUPIDPanoptic, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self._init_loss_regularization(loss_regularization)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head, \
            '3D EncoderDecoder Segmentor should have a decode_head'

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``."""
        self.decode_head = MODELS.build(decode_head)
        self.num_classes = self.decode_head.num_classes
        self.embed_dim = self.decode_head.embed_dim
        self.bandwidth = self.decode_head.meanshift_bandwidth
        self.clustering_method = self.decode_head.clustering_method
        self.kmeans_clusters = self.decode_head.kmeans_clusters

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
        x = self.backbone(batch_inputs)
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
        seg_logits = self.decode_head.predict(x, batch_input_metas,
                                              self.test_cfg)
        return seg_logits

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
        ins_logits = []  # save patch instance predictions

        for batch_idx in range(0, patch_points.shape[0], batch_size):
            batch_points = patch_points[batch_idx:batch_idx + batch_size]
            batch_points = batch_points.view(-1, num_points, feats_dim)
            # batch_seg_logit is of shape [B, num_classes, N]
            batch_seg_logit = self.encode_decode(batch_points,
                                                 [input_meta] * batch_size)
            batch_sem_logit = batch_seg_logit['sem_logit']
            batch_ins_logit = batch_seg_logit['ins_logit']
            batch_sem_logit = batch_sem_logit.transpose(1, 2).contiguous()
            batch_ins_logit = batch_ins_logit.transpose(1, 2).contiguous()
            seg_logits.append(batch_sem_logit.view(-1, self.num_classes))
            ins_logits.append(batch_ins_logit.view(-1, self.embed_dim))

        # aggregate per-point logits by indexing sum and dividing count
        seg_logits = torch.cat(seg_logits, dim=0)  # [K*N, num_classes]
        ins_logits = torch.cat(ins_logits, dim=0)  # [K*N, embed_dim]
        expand_patch_idxs_sem = patch_idxs.unsqueeze(1).repeat(1, self.num_classes)
        expand_patch_idxs_ins = patch_idxs.unsqueeze(1).repeat(1, self.embed_dim)
        preds_sem = point.new_zeros((point.shape[0], self.num_classes)).\
            scatter_add_(dim=0, index=expand_patch_idxs_sem, src=seg_logits)
        preds_ins = point.new_zeros((point.shape[0], self.embed_dim)).\
            scatter_add_(dim=0, index=expand_patch_idxs_ins, src=ins_logits)
        count_mat = torch.bincount(patch_idxs)
        preds_sem = preds_sem / count_mat[:, None]
        preds_ins = preds_ins / count_mat[:, None]

        # TODO: if rescale and voxelization segmentor

        # to [num_classes, K*N]
        return preds_sem.transpose(0, 1), preds_ins.transpose(0, 1)  

    # TODO: (michbaum) I think we want this one - at least for now with our small scenes
    def whole_inference(self, points: Tensor, batch_input_metas: List[dict],
                        rescale: bool) -> Tensor:
        """Inference with full scene (one forward pass without sliding)."""
        raise NotImplementedError('Not implemented yet.')
        seg_logit = self.encode_decode(points, batch_input_metas)
        # TODO: if rescale and voxelization segmentor
        return seg_logit

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
            Tensor: The output segmentation map.
        """
        assert self.test_cfg.mode in ['slide', 'whole']
        if self.test_cfg.mode == 'slide':
            seg_logits = [
                self.slide_inference(point, input_meta, rescale)
                for point, input_meta in zip(points, batch_input_metas)
            ]
            sem_logits = [seg_logit[0] for seg_logit in seg_logits]
            ins_logits = [seg_logit[1] for seg_logit in seg_logits]
            sem_logits = torch.stack(sem_logits, 0)
            ins_logits = torch.stack(ins_logits, 0)
        else:
            seg_logit = self.whole_inference(points, batch_input_metas,
                                             rescale)
        return sem_logits, ins_logits

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

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """
        # 3D segmentation requires per-point prediction, so it's impossible
        # to use down-sampling to get a batch of scenes with same num_points
        # therefore, we only support testing one scene every time
        sem_logits_list = []
        ins_logits_list = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)

        points = batch_inputs_dict['points']
        for point, input_meta in zip(points, batch_input_metas):
            sem_logits, ins_logits = self.inference(
                point.unsqueeze(0), [input_meta], rescale)
            sem_logits_list.append(sem_logits[0]) # (michbaum) Currently needed since we infere every batch individually
            ins_logits_list.append(ins_logits[0])

        return self.postprocess_result(sem_logits_list, ins_logits_list, batch_data_samples)

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

    def postprocess_result(self, sem_logits_list: List[Tensor],
                           ins_logits_list: List[Tensor],
                           batch_data_samples: SampleList) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Args:
            sem_logits_list (List[Tensor]): List of semantic segmentation results,
                sem_logits from model of each input point clouds sample.
            ins_logits_list (List[Tensor]): List of instance segmentation results,
                ins_logits from model of each input point clouds sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """

        for i in range(len(sem_logits_list)):
            seg_logits = sem_logits_list[i]
            seg_pred = seg_logits.argmax(dim=0)
            batch_data_samples[i].set_data({
                'pts_seg_logits':
                PointData(**{'pts_seg_logits': seg_logits}),
                'pred_pts_seg':
                PointData(**{'pts_semantic_mask': seg_pred})
            })

        # (michbaum) Instance segmentation post processing
        for j in range(len(ins_logits_list)):
            ins_logits = ins_logits_list[j]
            ins_pred = self.cluster_points(ins_logits)
            batch_data_samples[j].set_data({
                'pts_ins_logits':
                PointData(**{'pts_ins_logits': ins_logits}),
                'pred_pts_ins':
                PointData(**{'pts_instance_mask': ins_pred})
            })

        return batch_data_samples
    
    
    def cluster_points(self, embedding_logits):
        """ Compute clusters from embeddings using MeanShift clustering """
        
        # Initialize tensor to store the predicted instance labels
        predicted_ins_labels = -1 * torch.ones(embedding_logits.size(1), dtype=torch.int64)
        
        if self.clustering_method == 'meanshift':
            # Meanshift clustering for embeddings
            t_num_clusters, t_pre_ins_labels = self.meanshift_cluster(embedding_logits.transpose(0,1).detach().cpu().numpy(), self.bandwidth)
        elif self.clustering_method == 'kmeans':
            # kmeans cluster for embeddings
            t_pre_ins_labels, t_cluster_centers = self.kmeans_cluster(embedding_logits.transpose(0,1).detach().cpu().numpy())
        else:
            raise NotImplementedError(f"Clustering method {self.clustering_method} not implemented.")

        # Assign the clustered labels
        predicted_ins_labels = t_pre_ins_labels

        return predicted_ins_labels

    # (michbaum) Seems really slow and only finds around 7 clusters (with 19 objects)
    def meanshift_cluster(self, prediction, bandwidth):
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=4)
        ms.fit(prediction)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_ 	
        num_clusters = cluster_centers.shape[0]
     
        return num_clusters, torch.from_numpy(labels)

    # (michbaum) Probably needs some post-processing
    def kmeans_cluster(self, prediction):
        kmeans = KMeans(n_clusters=20, random_state=0)
        kmeans.fit(prediction)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        return torch.from_numpy(labels), torch.from_numpy(cluster_centers)


@MODELS.register_module()
class CUPIDPanopticMatching(Base3DSegmentor):
    """3D Encoder Decoder panoptic segmentor with matching postprocessing to match
    fragmented object instances in the scene.

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

    2. The ``predict`` method is used to predict segmentation results,
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
        matching_instance_class (int): The object class to instance match for.
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
                 matching_instance_class=2, # (michbaum) Which object class to instance match for
                 auxiliary_head: OptMultiConfig = None,
                 loss_regularization: OptMultiConfig = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 instance_overlap_threshold=0.5, # (michbaum) Threshold for instance matching
                 visualize_fails=False,
                 init_cfg: OptMultiConfig = None) -> None:
        super(CUPIDPanopticMatching, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.match_instances_class = matching_instance_class
        self.instance_overlap_threshold = instance_overlap_threshold
        self.visualize_fails = visualize_fails
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self._init_loss_regularization(loss_regularization)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head, \
            '3D EncoderDecoder Segmentor should have a decode_head'

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``."""
        self.decode_head = MODELS.build(decode_head)
        self.num_classes = self.decode_head.num_classes
        self.embed_dim = self.decode_head.embed_dim
        self.bandwidth = self.decode_head.meanshift_bandwidth
        self.clustering_method = self.decode_head.clustering_method
        self.kmeans_clusters = self.decode_head.kmeans_clusters

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
        x = self.backbone(batch_inputs)
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
        seg_logits = self.decode_head.predict(x, batch_input_metas,
                                              self.test_cfg)
        return seg_logits

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
        ins_logits = []  # save patch instance predictions

        for batch_idx in range(0, patch_points.shape[0], batch_size):
            batch_points = patch_points[batch_idx:batch_idx + batch_size]
            batch_points = batch_points.view(-1, num_points, feats_dim)
            # batch_seg_logit is of shape [B, num_classes, N]
            batch_seg_logit = self.encode_decode(batch_points,
                                                 [input_meta] * batch_size)
            batch_sem_logit = batch_seg_logit['sem_logit']
            batch_ins_logit = batch_seg_logit['ins_logit']
            batch_sem_logit = batch_sem_logit.transpose(1, 2).contiguous()
            batch_ins_logit = batch_ins_logit.transpose(1, 2).contiguous()
            seg_logits.append(batch_sem_logit.view(-1, self.num_classes))
            ins_logits.append(batch_ins_logit.view(-1, self.embed_dim))

        # aggregate per-point logits by indexing sum and dividing count
        seg_logits = torch.cat(seg_logits, dim=0)  # [K*N, num_classes]
        ins_logits = torch.cat(ins_logits, dim=0)  # [K*N, embed_dim]
        expand_patch_idxs_sem = patch_idxs.unsqueeze(1).repeat(1, self.num_classes)
        expand_patch_idxs_ins = patch_idxs.unsqueeze(1).repeat(1, self.embed_dim)
        preds_sem = point.new_zeros((point.shape[0], self.num_classes)).\
            scatter_add_(dim=0, index=expand_patch_idxs_sem, src=seg_logits)
        preds_ins = point.new_zeros((point.shape[0], self.embed_dim)).\
            scatter_add_(dim=0, index=expand_patch_idxs_ins, src=ins_logits)
        count_mat = torch.bincount(patch_idxs)
        preds_sem = preds_sem / count_mat[:, None]
        preds_ins = preds_ins / count_mat[:, None]

        # TODO: if rescale and voxelization segmentor

        # to [num_classes, K*N]
        return preds_sem.transpose(0, 1), preds_ins.transpose(0, 1)  

    # TODO: (michbaum) I think we want this one - at least for now with our small scenes
    def whole_inference(self, points: Tensor, batch_input_metas: List[dict],
                        rescale: bool) -> Tensor:
        """Inference with full scene (one forward pass without sliding)."""
        raise NotImplementedError('Not implemented yet.')
        seg_logit = self.encode_decode(points, batch_input_metas)
        # TODO: if rescale and voxelization segmentor
        return seg_logit

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
            Tensor: The output segmentation map.
        """
        assert self.test_cfg.mode in ['slide', 'whole']
        if self.test_cfg.mode == 'slide':
            seg_logits = [
                self.slide_inference(point, input_meta, rescale)
                for point, input_meta in zip(points, batch_input_metas)
            ]
            sem_logits = [seg_logit[0] for seg_logit in seg_logits]
            ins_logits = [seg_logit[1] for seg_logit in seg_logits]
            sem_logits = torch.stack(sem_logits, 0)
            ins_logits = torch.stack(ins_logits, 0)
        else:
            seg_logit = self.whole_inference(points, batch_input_metas,
                                             rescale)
        return sem_logits, ins_logits

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

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """
        # 3D segmentation requires per-point prediction, so it's impossible
        # to use down-sampling to get a batch of scenes with same num_points
        # therefore, we only support testing one scene every time
        sem_logits_list = []
        ins_logits_list = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)

        # (michbaum) Get the inference time
        # T1 = time.time()

        points = batch_inputs_dict['points']
        for point, input_meta in zip(points, batch_input_metas):
            sem_logits, ins_logits = self.inference(
                point.unsqueeze(0), [input_meta], rescale)
            sem_logits_list.append(sem_logits[0]) # (michbaum) Currently needed since we infere every batch individually
            ins_logits_list.append(ins_logits[0])

        # T2 = time.time()
        # (michbaum) Print the time in ms
        # print(f"Feature extraction time: {(T2-T1)*1000:.2f} ms")

        # (michbaum) For the matching mapping, we also need to know the instance priors
        assert [point.shape[1] >= 8 for point in points], "The input points must include the class and instance id as attributes"
        class_priors = [p[:, 6] for p in points]
        instance_priors = [p[:, 7] for p in points]
        priors = [torch.stack([class_prior, instance_prior], dim=1) for class_prior, instance_prior in zip(class_priors, instance_priors)]
        return self.postprocess_result(sem_logits_list, ins_logits_list, batch_data_samples, priors, points)

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

    def _extract_match_gt(self, instance_gt_mapping: Tensor, pcd_to_instance_mapping: Tensor, 
                          matching_indices: Tensor, distance_bools: Tensor, ignore_index: int) -> Tensor:
        """
        Build the ground truth matching mask from the metadata and the built pairs.

        Args:
            instance_gt_mapping (Tensor): Mapping from instance to ground truth class.
            pcd_to_instance_mapping (Tensor): Mapping from pointcloud to instance.
            matching_indices (Tensor): Indices of the matching pairs.
            distance_bools (Tensor): Whether the pairs are within the matching range.
            ignore_index (int): Index to ignore in the mask.

        Returns:
            Tensor: Mask per batch that indicates matches as 0, non-matches as 1 and ignored pairs as ignore_idx.
        """
        # (michbaum) Build the matching labels: 0 for match, 1 for non-match, ignore_index for pairs outside the matching range
        #            and for non-existent instance masks (that we filled up to a certain max size with -1 originally)
        match_gt = torch.ones_like(distance_bools[0], dtype=torch.long) * ignore_index
        for i, (pcd1, pcd2) in enumerate(matching_indices[0]):
            pcd1 = pcd1.item()
            pcd2 = pcd2.item()
            # (michbaum) Check if one of the pointclouds in the pair was a bogus fillup or if the
            #            pair was too far apart in euclidean space
            if pcd1 not in pcd_to_instance_mapping or pcd2 not in pcd_to_instance_mapping or not distance_bools[0][i]:
                match_gt[i] = ignore_index
            else:
                instance1 = pcd_to_instance_mapping[pcd1]
                instance2 = pcd_to_instance_mapping[pcd2]
                gt_instance1 = instance_gt_mapping[instance1]
                gt_instance2 = instance_gt_mapping[instance2]
                if gt_instance1 == gt_instance2:
                    match_gt[i] = 0
                else:
                    match_gt[i] = 1

        return match_gt

    
    def visualize_failures(self, gt_matching_mask, pred_pair_matching_mask, pointcloud, 
                           feature_indices, pcd_to_instance_mapping):
        """
        Visualize false negative matches (missed matches) and false positive matches (wrong matches).

        Args:
            gt_matching_masks (list[Tensor]): Ground truth matching masks.
            pred_pair_matching_masks (list[Tensor]): Predicted matching mask.
            pointcloud_pairs_list (Tensor): Associated pointcloud.
        """
        from mmdet3d.visualization import Det3DLocalVisualizer

        # We get the indices of the false negatives (gt 0 but pred 1)
        false_negatives = (gt_matching_mask == 0) & (pred_pair_matching_mask.cpu().numpy() == 1)
        false_neg_indices = torch.nonzero(torch.from_numpy(false_negatives[0]))
        
        # We get the indices of the false positives (gt 1 but pred 0)
        false_positives = (gt_matching_mask == 1) & (pred_pair_matching_mask.cpu().numpy() == 0)
        false_pos_indices = torch.nonzero(torch.from_numpy(false_positives[0]))

        # We visualize the associated pointcloud pairs
        for i in false_neg_indices:
            print("Looking at false negatives")
            # (michbaum) Extract the partial pointclouds
            label1, label2 = feature_indices[0][i.item()]
            label1 = pcd_to_instance_mapping[label1.item()]
            label2 = pcd_to_instance_mapping[label2.item()]
            pc_1 = pointcloud[pointcloud[:, 7] == label1]
            pc_2 = pointcloud[pointcloud[:, 7] == label2]
            # (michbaum) Combine the pointclouds
            pc = torch.cat([pc_1, pc_2], dim=0)
            visualizer = Det3DLocalVisualizer()
            visualizer.set_points(np.asarray(pc.cpu()), pcd_mode=2, vis_mode='add', mode='xyzrgb')
            visualizer.show()
            visualizer._clear_o3d_vis()

        # # We visualize the associated pointcloud pairs
        for j in false_pos_indices:
            print("Looking at false positives")
            label1, label2 = feature_indices[0][j.item()]
            label1 = pcd_to_instance_mapping[label1.item()]
            label2 = pcd_to_instance_mapping[label2.item()]
            pc_1 = pointcloud[pointcloud[:, 7] == label1]
            pc_2 = pointcloud[pointcloud[:, 7] == label2]
            # (michbaum) Combine the pointclouds
            pc = torch.cat([pc_1, pc_2], dim=0)
            visualizer = Det3DLocalVisualizer()
            visualizer.set_points(np.asarray(pc.cpu()), pcd_mode=2, vis_mode='add', mode='xyzrgb')
            visualizer.show()
            visualizer._clear_o3d_vis()

    def postprocess_result(self, sem_logits_list: List[Tensor],
                           ins_logits_list: List[Tensor],
                           batch_data_samples: SampleList,
                           priors: list[Tensor], points: list[Tensor]) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Args:
            sem_logits_list (List[Tensor]): List of semantic segmentation results,
                sem_logits from model of each input point clouds sample.
            ins_logits_list (List[Tensor]): List of instance segmentation results,
                ins_logits from model of each input point clouds sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            priors (list[Tensor]): The instance priors for each point cloud point.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """

        for i in range(len(sem_logits_list)):
            seg_logits = sem_logits_list[i]
            seg_pred = seg_logits.argmax(dim=0)
            batch_data_samples[i].set_data({
                'pts_seg_logits':
                PointData(**{'pts_seg_logits': seg_logits}),
                'pred_pts_seg':
                PointData(**{'pts_semantic_mask': seg_pred})
            })

        # (michbaum) Get the clustering time
        # T1 = time.time()
        # (michbaum) Instance segmentation post processing
        for j in range(len(ins_logits_list)):
            match_logits = ins_logits_list[j]
            ins_pred = self.cluster_points(match_logits)
            batch_data_samples[j].set_data({
                'pts_ins_logits':
                PointData(**{'pts_ins_logits': match_logits}),
                'pred_pts_ins':
                PointData(**{'pts_instance_mask': ins_pred})
            })
        # T2 = time.time()
        # (michbaum) Print the time in ms
        # print(f"Clustering time: {(T2-T1)*1000:.2f} ms")

        # (michbaum) Get the matching time
        # T3 = time.time()
        # (michbaum) Transform instance predictions into instance matchings
        for k in range(len(batch_data_samples)):
            # (michbaum) Build prediction for all instance pairs and populate
            #            instance mapping dicts for evaluation
            match_logits, feature_indices, distance_bools, \
                 pcd_to_instance_mapping, instance_gt_mapping = self.match_instances(batch_data_samples[k], priors[k])
            match_pred = match_logits.argmax(dim=1)


            # (michbaum) Failure visualization
            if self.visualize_fails:
                gt_matching_mask = np.asarray(self._extract_match_gt(instance_gt_mapping,
                                                            pcd_to_instance_mapping,
                                                            feature_indices,
                                                            distance_bools,
                                                            2).cpu())

                pointcloud = points[k]
                self.visualize_failures(gt_matching_mask, match_pred, pointcloud, feature_indices, pcd_to_instance_mapping)

            batch_data_samples[k].set_data({
                'pair_matching_logits':
                PointData(**{'pair_matching_logits': match_logits}),
                'pred_pair_matching':
                PointData(**{'pred_pair_matching': match_pred}),
                'feature_pair_indices':
                PointData(**{'feature_pair_indices': feature_indices}),
                'pair_distance_bools':
                PointData(**{'pair_distance_bools': distance_bools}) # (michbaum) True for all pairs since it doesn't apply
            })
        # T4 = time.time()
        # (michbaum) Print the time in ms
        # print(f"Matching time: {(T4-T3)*1000:.2f} ms")
            
        # TODO: (michbaum) Implement
        # if self.postprocess_matches:
        #     # (michbaum) Postprocess the results if desired. This makes sure that only
        #     #            1 match per instance is predicted, either greedily or with
        #     #            the Hungarian algorithm predicting the 'optimal' matches
        #     assert self.postprocess_strategy in ['greedy', 'hungarian'], \
        #         "Invalid postprocessing strategy. Choose either 'greedy' or 'hungarian'."
            
        #     if self.postprocess_strategy == 'greedy':
        #         self.greedy_postprocess(match_logits_list, feature_indices_list, batch_data_samples)
        #     elif self.postprocess_strategy == 'hungarian':
        #         self.hungarian_postprocess(match_logits_list, feature_indices_list, batch_data_samples)

        return batch_data_samples
    
    def match_instances(self, data_sample, priors) -> tuple[Tensor, Tensor, Tensor]:
        """
        Transforms instance mask predictions into instance matching results.

        Args:
            ins_pred (Tensor): Instance ID prediction for every point in the pointcloud
            data_sample (SampleList): The det3d data sample.
            priors (Tensor): The class and instance priors for each point cloud point.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Instance matching results, feature indices and distance bools.

            Also populates the data_sample['eval_ann_info'] dictionary with the instance_gt_mapping and
            the pcd_to_instance_mapping.
        """
        pred_pts_instance_mask = data_sample.pred_pts_ins.pts_instance_mask # predicted instances
        gt_pts_instance_mask = torch.from_numpy(data_sample.eval_ann_info['pts_instance_mask']).to(pred_pts_instance_mask.device) # ground truth instances

        prior_pts_instance_mask = priors[:, 1] # prior instances
        prior_pts_class_mask = priors[:, 0] # prior classes

        # (michbaum) We currently only support matching a single object class, so we filter here
        mask = prior_pts_class_mask == self.match_instances_class
        prior_pts_instance_mask = prior_pts_instance_mask[mask]
        pred_pts_instance_mask = pred_pts_instance_mask[mask]
        gt_pts_instance_mask = gt_pts_instance_mask[mask]

        # (michbaum) We build a potential matching pair for all distinct pairs of instance labels
        #            in our priors, save the indices in the feature_indices tensor and an according
        #            mapping to the instance priors in data_sample.eval_ann_info['pcd_to_instance_mapping']
        #            as well as the instance_gt_mapping.
        prior_ins_labels = torch.unique(prior_pts_instance_mask)
        num_priors = prior_ins_labels.size(0)

        # (michbaum) Count the number of shared instance points between all prior pairs as our matching
        #            score. Then we can once again use Greedy or Hungarian matching to find the best
        #            matching pairs. 
        
        # Create feature indices for all pairs
        i_indices, j_indices = torch.triu_indices(num_priors, num_priors, offset=1)
        
        # Calculate matching scores
        matching_scores = torch.zeros(i_indices.size(0), device=pred_pts_instance_mask.device)
        
        for i in range(i_indices.size(0)):
            prior1, prior2 = prior_ins_labels[i_indices[i]], prior_ins_labels[j_indices[i]]
            mask1 = prior_pts_instance_mask == prior1
            mask2 = prior_pts_instance_mask == prior2
            
            # (michbaum) For every predicted instance label/cluster, we calculate the number of shared points
            #            with the two prior instances. This is our matching score (accumulated over multiple
            #            masks since we have more clusters then objects due to KMeans clustering use).
            for pred_label in torch.unique(pred_pts_instance_mask):
                pred_mask = pred_pts_instance_mask == pred_label
                score = min(torch.sum(mask1 & pred_mask)/torch.sum(mask1), torch.sum(mask2 & pred_mask)/torch.sum(mask2))
                matching_scores[i] += score
        
        # (michbaum) Build the logits tensor
        # (michbaum) Score thresholding: we want at least x% of the instances to overlap
        threshold = self.instance_overlap_threshold
        matching_logits = threshold * torch.ones(2, matching_scores.size(0), device=matching_scores.device)
        matching_logits[0] = matching_scores
        # matching_scores = torch.where(matching_scores > threshold, matching_scores, torch.zeros_like(matching_scores))

        matching_logits.unsqueeze_(0) # Needed for eval

        # Create pcd_to_instance_mapping
        pcd_to_instance_mapping = {i: int(label.item()) for i, label in enumerate(prior_ins_labels)}
        
        # Create instance_gt_mapping
        instance_gt_mapping = {}
        for prior_label in prior_ins_labels:
            prior_mask = prior_pts_instance_mask == prior_label
            gt_labels, gt_counts = torch.unique(gt_pts_instance_mask[prior_mask], return_counts=True)
            # (michbaum) We choose the ground truth label with the most shared points as the prior label,
            #            could also use torch.mode
            if gt_labels.size(0) > 0:
                instance_gt_mapping[int(prior_label.item())] = int(gt_labels[torch.argmax(gt_counts)].item())
        
        # Populate data_sample['eval_ann_info']
        data_sample.eval_ann_info['pcd_to_instance_mapping'] = pcd_to_instance_mapping
        data_sample.eval_ann_info['instance_gt_mapping'] = instance_gt_mapping
        
        feature_indices = torch.stack([i_indices, j_indices], dim=0).transpose(0, 1).unsqueeze(0)
        # Distance bools are not applicable, so we set them to True
        distance_bools = torch.ones(feature_indices.size(1), dtype=torch.bool, device=pred_pts_instance_mask.device).unsqueeze(0)

        # (michbaum) Check that our mapping is truly correct
        check = [instance_gt_mapping[pcd_to_instance_mapping[i.item()]] == instance_gt_mapping[pcd_to_instance_mapping[j.item()]] for i, j in feature_indices[0]]
    
        return matching_logits, feature_indices, distance_bools, pcd_to_instance_mapping, instance_gt_mapping

    
    def cluster_points(self, embedding_logits):
        """ Compute clusters from embeddings using MeanShift clustering """
        
        # Initialize tensor to store the predicted instance labels
        predicted_ins_labels = -1 * torch.ones(embedding_logits.size(1), dtype=torch.int64)
        
        if self.clustering_method == 'meanshift':
            # Meanshift clustering for embeddings
            t_num_clusters, t_pre_ins_labels = self.meanshift_cluster(embedding_logits.transpose(0,1).detach().cpu().numpy())
        elif self.clustering_method == 'kmeans':
            # kmeans cluster for embeddings
            t_pre_ins_labels, t_cluster_centers = self.kmeans_cluster(embedding_logits.transpose(0,1).detach().cpu().numpy())
        else:
            raise NotImplementedError(f"Clustering method {self.clustering_method} not implemented.")

        # Assign the clustered labels
        predicted_ins_labels = t_pre_ins_labels.to(embedding_logits.device)

        return predicted_ins_labels

    # (michbaum) Seems really slow and only finds around 7 clusters (with 19 objects)
    def meanshift_cluster(self, prediction):
        ms = MeanShift(bandwidth=self.bandwidth, bin_seeding=True, n_jobs=4)
        ms.fit(prediction)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_ 	
        num_clusters = cluster_centers.shape[0]
     
        return num_clusters, torch.from_numpy(labels)

    # (michbaum) Probably needs some post-processing
    def kmeans_cluster(self, prediction):
        kmeans = KMeans(n_clusters=self.kmeans_clusters, random_state=0)
        kmeans.fit(prediction)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        return torch.from_numpy(labels), torch.from_numpy(cluster_centers)