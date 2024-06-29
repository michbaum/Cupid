# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModule
from torch import nn as nn

from mmdet3d.models.layers.pointnet_modules import PointFPModule
from mmdet3d.registry import MODELS

import torch


# TODO: (michbaum) Adapt
@MODELS.register_module()
class CUPIDNeck(BaseModule):
    r"""Feature vector Matching Module used in CUPID.

    Args:
        matching_range (float): Euclidean distance within which feature vectors
            will be matched. If not given, all possible pairs will be built.
        use_xyz (bool): Whether to use xyz as a part of features.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self, matching_range=None, use_xyz=False, init_cfg=None):
        super(CUPIDNeck, self).__init__(init_cfg=init_cfg)

        self.matching_range = matching_range
        self.use_xyz = use_xyz

    def _extract_input(self, feat_dict):
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone, which may contain
                the following keys and values:

                - sa_xyz (list[torch.Tensor]): Points of each sa module
                    in shape (N, 3).
                - sa_features (list[torch.Tensor]): Output features of
                    each sa module in shape (N, M).

        Returns:
            list[torch.Tensor]: Coordinates of multiple levels of points.
            list[torch.Tensor]: Features of multiple levels of points.
        """
        sa_xyz = feat_dict['sa_xyz']
        sa_features = feat_dict['sa_features']
        assert len(sa_xyz) == len(sa_features)

        return sa_xyz, sa_features
    
    def _build_pairs(self, xyz, features):
        """
        Builds all possible pairs of feature vectors from the instances.

        Args:
            xyz (torch.tensor): Tensor containing the feature positions with shape [B, N, 3]
            features (torch.tensor): _description_
        """
        # Get the size parameters
        batch_size, num_vectors, _ = features.shape

        # Concatenate the xyz coordinates if wanted
        if self.use_xyz:
            features_xyz = torch.cat((xyz, features), dim=-1)
        else:
            features_xyz = features

        # Create a meshgrid of indices for the pairs
        i_indices, j_indices = torch.triu_indices(num_vectors, num_vectors, offset=1)

        # Expand the indices to cover all batches
        i_indices = i_indices.unsqueeze(0).expand(batch_size, -1)
        j_indices = j_indices.unsqueeze(0).expand(batch_size, -1)

        # Gather the pairs
        first_vectors = features_xyz[torch.arange(batch_size)[:, None], i_indices]
        second_vectors = features_xyz[torch.arange(batch_size)[:, None], j_indices]

        # Concatenate the pairs along the feature dimension
        concatenated_pairs = torch.cat((first_vectors, second_vectors), dim=-1)

        if self.matching_range is not None:
            # Calculate Euclidean distances between xyz coordinates of the pairs
            first_xyz = xyz[torch.arange(batch_size)[:, None], i_indices]
            second_xyz = xyz[torch.arange(batch_size)[:, None], j_indices]
            distances = torch.norm(first_xyz - second_xyz, dim=-1)

            # Compare distances with the threshold
            distance_bools = distances < self.matching_range
        else:
            distance_bools = torch.ones_like(i_indices, dtype=torch.bool)

        # Save the index pairs
        index_pairs = torch.stack((i_indices, j_indices), dim=2)  # shape: (batch_size, num_pairs, 2)

        return concatenated_pairs, index_pairs, distance_bools


    def forward(self, feat_dict):
        """Forward pass. Builds feature pairs and concatenates their features
        for matching down the line.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            dict[str, torch.Tensor]: Outputs of the Neck.

                - stacked_features (torch.Tensor): Concatenated features of
                    all pairs of instances.
                - feature_indices (list[tuples]): Indices of the features
                    that were concatenated.
        """
        sa_xyz, sa_features = self._extract_input(feat_dict)

        fp_feature = sa_features[-1]
        fp_xyz = sa_xyz[-1]

        # (michbaum) For every batch, we build all possible pairs of instances
        #            (even the ones that have been filled up and don't belong to masks)

        # (michbaum) Additionally, we save the indices of the features that were concatenated

        # (michbaum) Very naive matching with (50, 2) complexity
        concatenated_pairs, index_pairs, distance_bools = self._build_pairs(fp_xyz, fp_feature)

        ret = dict(stacked_features=concatenated_pairs, feature_indices=index_pairs, distance_bools=distance_bools)
        return ret
