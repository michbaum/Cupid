# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from typing import Dict, Optional, Sequence

import mmcv
import numpy as np
import torch
from torch import Tensor
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet3d.evaluation import match_eval
from mmdet3d.registry import METRICS


@METRICS.register_module()
class MatchMetric(BaseMetric):
    """3D instance matching evaluation metric.

    Args:
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
        pklfile_prefix (str, optional): The prefix of pkl files, including
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        submission_prefix (str, optional): The prefix of submission data.
            If not specified, the submission data will not be generated.
            Default: None.
    """

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 **kwargs):
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        super(MatchMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            pred_pair_matching = data_sample['pred_pair_matching']
            feature_pair_indices = data_sample['feature_pair_indices']
            pair_distance_bools = data_sample['pair_distance_bools']
            eval_ann_info = data_sample['eval_ann_info']
            # TODO: (michbaum) Add the matching_indices and within_range stuff
            cpu_pred_3d = dict()
            for k, v in pred_pair_matching.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu').numpy()
                else:
                    cpu_pred_3d[k] = v
            self.results.append((eval_ann_info, cpu_pred_3d, feature_pair_indices, pair_distance_bools))

    # (michbaum) Not used by us and not adapted
    def format_results(self, results):
        r"""Format the results to txt file. Refer to `ScanNet documentation
        <http://kaldir.vc.in.tum.de/scannet_benchmark/documentation>`_.

        Args:
            outputs (list[dict]): Testing results of the dataset.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving submission
                files when ``submission_prefix`` is not specified.
        """

        submission_prefix = self.submission_prefix
        if submission_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            submission_prefix = osp.join(tmp_dir.name, 'results')
        mmcv.mkdir_or_exist(submission_prefix)
        ignore_index = self.dataset_meta['ignore_index']
        # need to map network output to original label idx
        cat2label = np.zeros(len(self.dataset_meta['label2cat'])).astype(
            np.int64)
        for original_label, output_idx in self.dataset_meta['label2cat'].items(
        ):
            if output_idx != ignore_index:
                cat2label[output_idx] = original_label

        for i, (eval_ann, result) in enumerate(results):
            sample_idx = eval_ann['point_cloud']['lidar_idx']
            pred_sem_mask = result['semantic_mask'].numpy().astype(np.int64)
            pred_label = cat2label[pred_sem_mask]
            curr_file = f'{submission_prefix}/{sample_idx}.txt'
            np.savetxt(curr_file, pred_label, fmt='%d')


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


    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.submission_prefix:
            raise NotImplementedError(
                'Submission is not supported for MatchMetric.')
            self.format_results(results)
            return None

        label2cat = self.dataset_meta['label2cat'] # (michbaum) Not used by us
        label2cat = {0: 'match', 1: 'no_match'}
        ignore_index = self.dataset_meta['ignore_index'] # (michbaum) Not used by us (is for class semantic segmentation)
        ignore_index = 2

        gt_matching_masks = []
        pred_pair_matching_masks = []

        # TODO: (michbaum) Need to build a gt mask just like in the loss function
        for eval_ann, single_pred_results, single_pair_indices, single_distance_bools in results:
            instance_gt_mapping = eval_ann['instance_gt_mapping']
            pcd_to_instance_mapping = eval_ann['pcd_to_instance_mapping']
            single_pair_indices = single_pair_indices['feature_pair_indices']
            single_distance_bools = single_distance_bools['pair_distance_bools']
            gt_matching_masks.append(np.asarray(self._extract_match_gt(instance_gt_mapping,
                                                            pcd_to_instance_mapping,
                                                            single_pair_indices,
                                                            single_distance_bools,
                                                            ignore_index))) # TODO: (michbaum) Save it in here
            pred_pair_matching_masks.append(
                single_pred_results['pred_pair_matching'][0])

        ret_dict = match_eval(
            gt_matching_masks,
            pred_pair_matching_masks,
            label2cat,
            ignore_index,
            logger=logger)

        return ret_dict
