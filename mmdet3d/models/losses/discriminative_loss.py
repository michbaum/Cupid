# Michael Baumgartner: Adapted from:
# https://github.com/prs-eth/PanopticSegForMobileMappingPointClouds/blob/12ead8d8a4d83e0025d959b40fd5d490d81dd787/torch_points3d/core/losses/panoptic_losses.py#L48C1-L188C38
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch import scatter
from torch import nn as nn
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss

from mmdet3d.registry import MODELS



@MODELS.register_module()
class DiscriminativeLoss(nn.Module):
    """Calculate Discriminative Loss of an instance labeling.

    Args:
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        loss_weight (float): Weight of loss. Defaults to l.0.
    """

    def __init__(self,
                 ignore_index: Optional[int] = None,
                 loss_weight: float = 1.0,
                 ) -> None:
        super(DiscriminativeLoss, self).__init__()

        self.ignore_index = -100 if ignore_index is None else ignore_index
        self.loss_weight = loss_weight

    def forward(self,
        embedding_logits: torch.Tensor,
        semantic_labels: torch.Tensor,
        instance_labels: torch.Tensor,
        feature_dim,
        ignore_idx: Optional[int] = None,
    ):
        ignore_idx = self.ignore_index if ignore_idx is None else ignore_idx
        loss = []
        loss_var = []
        loss_dist = []
        loss_reg = []
        batch_size = embedding_logits.size()[0]
        for s in range(batch_size):
            sample_gt_class = semantic_labels[s]
            sample_gt_instances = instance_labels[s]
            sample_embed_logits = embedding_logits[s]
            sample_loss, sample_loss_var, sample_loss_dist, sample_loss_reg = self.discriminative_loss_single(sample_embed_logits, 
                                                                                                         sample_gt_class, 
                                                                                                         sample_gt_instances, 
                                                                                                         feature_dim, 
                                                                                                         ignore_idx)
            loss.append(sample_loss)
            loss_var.append(sample_loss_var)
            loss_dist.append(sample_loss_dist)
            loss_reg.append(sample_loss_reg)
        loss = torch.stack(loss)
        loss_var = torch.stack(loss_var)
        loss_dist = torch.stack(loss_dist)
        loss_reg = torch.stack(loss_reg)
        return {"ins_loss": torch.mean(loss), "ins_var_contribution": torch.mean(loss_var), "ins_dist_contribution": torch.mean(loss_dist), "ins_reg_contribution": torch.mean(loss_reg)}
        #return torch.mean(loss), torch.mean(loss_var), torch.mean(loss_dist), torch.mean(loss_reg)


    def discriminative_loss_single(self,
        prediction,
        correct_class,
        correct_instance,
        feature_dim,
        ignore_idx,
        delta_v = 0.5,
        delta_d = 1.5,
        param_var = 1.,
        param_dist = 1.,
        param_reg = 0.001,
    ):

        ''' Discriminative loss for a single prediction/label pair.
            :param prediction: inference of network
            :param correct_class: class label
            :param correct_instance: instance label
            :feature_dim: feature dimension of prediction
            :ignore_idx: ignore index -> ignore points with this class label from the loss computation
            :param delta_v: cutoff variance distance
            :param delta_d: curoff cluster distance
            :param param_var: weight for intra cluster variance
            :param param_dist: weight for inter cluster distances
            :param param_reg: weight regularization
        '''
        ### Reshape so pixels are aligned along a vector [embeds, pixels] -> [pixels, embeds]
        # (michbaum) Pretty sure this is wrong in our case, simply transpose
        # reshaped_pred = torch.reshape(prediction, (-1, feature_dim))
        reshaped_pred = prediction.transpose(0, 1)

        # (michbaum)
        # Filter out points and corresponding instance labels with ignore_idx class label
        reshaped_pred = reshaped_pred[correct_class != ignore_idx]
        correct_instance = correct_instance[correct_class != ignore_idx]

        # TODO: (michbaum) Change instance IDs to be exclusive over different classes in the batch
        # Currently hardcoded since we only have 1 table but could be made smarter
        # change the instance ID of all tables (class ID 1) to 0 (because right now they're all 1)
        correct_instance[correct_class[correct_class!=0] == 1] = 0

        ### Count instances
        unique_labels, unique_id, counts = torch.unique(correct_instance, return_inverse=True, return_counts=True)
        #counts = tf.cast(counts, tf.float32)
        
        num_instances = unique_labels.shape[0]
        
        # (michbaum) Create a new tensor to hold the summed embeddings
        segmented_sum = torch.zeros((num_instances, feature_dim), device=reshaped_pred.device)

        segmented_sum.scatter_add_(0, unique_id.unsqueeze(1).expand(-1, feature_dim), reshaped_pred)

        # mu = torch.div(segmented_sum, (torch.reshape(counts, (-1, 1)) + 1e-8 ))
        mu = torch.div(segmented_sum, (counts.unsqueeze(1) + 1e-8))
        # unique_id_t = unique_id.unsqueeze(1)
        
        # unique_id_t = unique_id_t.expand(unique_id_t.size()[0], mu.size()[-1])
        # mu_expand = torch.gather(mu, 0, unique_id_t)
        mu_expand = mu[unique_id]

        ### Calculate l_var
        #distance = tf.norm(tf.subtract(mu_expand, reshaped_pred), axis=1)
        #tmp_distance = tf.subtract(reshaped_pred, mu_expand)
        tmp_distance = reshaped_pred - mu_expand
        distance = torch.norm(tmp_distance, p=1, dim=1) # (michbaum) Could also use p=2 for L2 norm
        # distance = torch.subtract(distance, delta_v)
        # distance = torch.clip(distance, min=0.)
        distance = torch.clamp(distance - delta_v, min=0.)
        distance = torch.square(distance)
        # l_var = torch.zeros((num_instances, feature_dim)).cuda()
        l_var = torch.zeros(num_instances, device=reshaped_pred.device)
        # l_var.scatter_add_(0, unique_id.unsqueeze(1).expand(-1, feature_dim), distance)
        l_var.scatter_add_(0, unique_id, distance)
        l_var = torch.div(l_var, counts + 1e-8)
        l_var = torch.sum(l_var)
        l_var = torch.div(l_var, float(num_instances))

        ### Calculate l_dist

        # Get distance for each pair of clusters like this:
        #   mu_1 - mu_1
        #   mu_2 - mu_1
        #   mu_3 - mu_1
        #   mu_1 - mu_2
        #   mu_2 - mu_2
        #   mu_3 - mu_2
        #   mu_1 - mu_3
        #   mu_2 - mu_3
        #   mu_3 - mu_3

        mu_interleaved_rep = mu.repeat(num_instances, 1)
        mu_band_rep = mu.repeat(1, num_instances)
        mu_band_rep = torch.reshape(mu_band_rep, (num_instances * num_instances, feature_dim))

        mu_diff = torch.subtract(mu_band_rep, mu_interleaved_rep)
        # Filter out zeros from same cluster subtraction
        eye = torch.eye(num_instances)
        #zero = torch.zeros(1, dtype=torch.float32)
        diff_cluster_mask = torch.eq(eye, 0)
        diff_cluster_mask = torch.reshape(diff_cluster_mask, (-1,))
        mu_diff_bool = mu_diff[diff_cluster_mask]
        #intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff),axis=1)
        #zero_vector = tf.zeros(1, dtype=tf.float32)
        #bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
        #mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)

        mu_norm = torch.norm(mu_diff_bool, p=1, dim=1)
        mu_norm = torch.subtract(torch.mul(delta_d, 2.0), mu_norm)
        mu_norm = torch.clip(mu_norm, min=0.)
        mu_norm = torch.square(mu_norm)

        l_dist = torch.mean(mu_norm)
        
        if num_instances==1:
            l_dist = torch.tensor(0).cuda()
        ### Calculate l_reg
        l_reg = torch.mean(torch.norm(mu, p=1, dim=1))

        if num_instances==0:
            l_var = torch.tensor(0).cuda()
            l_dist = torch.tensor(0).cuda()
            l_reg = torch.tensor(0).cuda()
        
        # param_scale = 1.
        param_scale = self.loss_weight
        l_var = param_var * l_var
        l_dist = param_dist * l_dist
        l_reg = param_reg * l_reg

        loss = param_scale * (l_var + l_dist + l_reg)

        #if torch.is_tensor(loss):
        #    loss = loss.item()
        #if torch.is_tensor(l_var):
        #    l_var = l_var.item()
        #if torch.is_tensor(l_dist):
        #    l_dist = l_dist.item()
        #if torch.is_tensor(l_reg):
        #    l_reg = l_reg.item()

        return loss, l_var, l_dist, l_reg

        
