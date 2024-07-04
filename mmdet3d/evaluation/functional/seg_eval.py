# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmengine.logging import print_log
from terminaltables import AsciiTable


def fast_hist(preds, labels, num_classes):
    """Compute the confusion matrix for every batch.

    Args:
        preds (np.ndarray):  Prediction labels of points with shape of
        (num_points, ).
        labels (np.ndarray): Ground truth labels of points with shape of
        (num_points, ).
        num_classes (int): number of classes

    Returns:
        np.ndarray: Calculated confusion matrix.
    """

    k = (labels >= 0) & (labels < num_classes)
    bin_count = np.bincount(
        num_classes * labels[k].astype(int) + preds[k],
        minlength=num_classes**2)
    return bin_count[:num_classes**2].reshape(num_classes, num_classes)


def per_class_iou(hist):
    """Compute the per class iou.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        np.ndarray: Calculated per class iou
    """

    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def get_acc(hist):
    """Compute the overall accuracy.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        float: Calculated overall acc
    """

    return np.diag(hist).sum() / hist.sum()


def get_acc_cls(hist):
    """Compute the class average accuracy.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        float: Calculated class average acc
    """

    return np.nanmean(np.diag(hist) / hist.sum(axis=1))


def seg_eval(gt_labels, seg_preds, label2cat, ignore_index, logger=None):
    """Semantic Segmentation  Evaluation.

    Evaluate the result of the Semantic Segmentation.

    Args:
        gt_labels (list[torch.Tensor]): Ground truth labels.
        seg_preds  (list[torch.Tensor]): Predictions.
        label2cat (dict): Map from label to category name.
        ignore_index (int): Index that will be ignored in evaluation.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Returns:
        dict[str, float]: Dict of results.
    """
    assert len(seg_preds) == len(gt_labels)
    num_classes = len(label2cat)

    hist_list = []
    for i in range(len(gt_labels)):
        gt_seg = gt_labels[i].astype(np.int64)
        pred_seg = seg_preds[i].astype(np.int64)

        # filter out ignored points
        pred_seg[gt_seg == ignore_index] = -1
        gt_seg[gt_seg == ignore_index] = -1

        # calculate one instance result
        hist_list.append(fast_hist(pred_seg, gt_seg, num_classes))

    iou = per_class_iou(sum(hist_list))
    # if ignore_index is in iou, replace it with nan
    if ignore_index < len(iou):
        iou[ignore_index] = np.nan
    miou = np.nanmean(iou)
    acc = get_acc(sum(hist_list))
    acc_cls = get_acc_cls(sum(hist_list))

    header = ['classes']
    for i in range(len(label2cat)):
        header.append(label2cat[i])
    header.extend(['miou', 'acc', 'acc_cls'])

    ret_dict = dict()
    table_columns = [['results']]
    for i in range(len(label2cat)):
        ret_dict[label2cat[i]] = float(iou[i])
        table_columns.append([f'{iou[i]:.4f}'])
    ret_dict['miou'] = float(miou)
    ret_dict['acc'] = float(acc)
    ret_dict['acc_cls'] = float(acc_cls)

    table_columns.append([f'{miou:.4f}'])
    table_columns.append([f'{acc:.4f}'])
    table_columns.append([f'{acc_cls:.4f}'])

    table_data = [header]
    table_rows = list(zip(*table_columns))
    table_data += table_rows
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)

    return ret_dict

def match_eval(gt_labels, match_preds, label2cat, ignore_index, logger=None):
    """Instance Matching Evaluation.

    Evaluate the result of the Instance Matching.

    Args:
        gt_labels (list[torch.Tensor]): Ground truth labels. 0 match, 1 no_match. Ignore index is 2 typically.
        seg_preds  (list[torch.Tensor]): Predictions.
        label2cat (dict): Map from label to category name.
        ignore_index (int): Index that will be ignored in evaluation.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Returns:
        dict[str, float]: Dict of results.
    """
    # TODO: (michabum) Change accordingly
    assert len(match_preds) == len(gt_labels)
    num_classes = len(label2cat)

    hist_list = []
    for i in range(len(gt_labels)):
        gt_match = gt_labels[i].astype(np.int64)
        pred_match = match_preds[i].astype(np.int64)

        # filter out ignored points
        pred_match[gt_match == ignore_index] = -1
        gt_match[gt_match == ignore_index] = -1

        # calculate one instance result
        hist_list.append(fast_hist(pred_match, gt_match, num_classes))

    histogram = sum(hist_list)

    # Precision - TP / (TP + FP)
    precision = np.diag(histogram)[0] / histogram.sum(0)[0]

    # Sensitivity/Match recall - TP / (TP + FN)
    sensitivity = np.diag(histogram)[0] / histogram.sum(1)[0]

    # Specificity/No_Match recall - TN / (TN + FP)
    specificity = np.diag(histogram)[1] / histogram.sum(1)[1]

    # F1 Score - 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    # iou = per_class_iou(sum(hist_list))
    # # if ignore_index is in iou, replace it with nan
    # if ignore_index < len(iou):
    #     iou[ignore_index] = np.nan
    # miou = np.nanmean(iou)
    acc = get_acc(sum(hist_list))
    acc_cls = get_acc_cls(sum(hist_list))

    header = ['metrics']
    # for i in range(len(label2cat)):
    #     header.append(label2cat[i])
    header.extend(['precision', 'sensitivity', 'specificity', 'f1-score', 'acc', 'acc_cls'])

    ret_dict = dict()
    table_columns = [['results']]
    # for i in range(len(label2cat)):
    #     ret_dict[label2cat[i]] = float(iou[i])
    #     table_columns.append([f'{iou[i]:.4f}'])
    # ret_dict['miou'] = float(miou)
    ret_dict['precision'] = float(precision)
    ret_dict['sensitivity'] = float(sensitivity)
    ret_dict['specificity'] = float(specificity)
    ret_dict['f1_score'] = float(f1_score)
    ret_dict['acc'] = float(acc)
    ret_dict['acc_cls'] = float(acc_cls)

    # table_columns.append([f'{miou:.4f}'])
    table_columns.append([f'{precision:.4f}'])
    table_columns.append([f'{sensitivity:.4f}'])
    table_columns.append([f'{specificity:.4f}'])
    table_columns.append([f'{f1_score:.4f}'])
    table_columns.append([f'{acc:.4f}'])
    table_columns.append([f'{acc_cls:.4f}'])

    table_data = [header]
    table_rows = list(zip(*table_columns))
    table_data += table_rows
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)

    return ret_dict

def instance_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union != 0 else 0

def cupid_panoptic_seg_eval(gt_semantic_masks, gt_instance_masks, pred_semantic_masks, pred_instance_masks, label2cat, ignore_index, logger=None):
    """Panoptic Segmentation  Evaluation.

    Evaluate the result of the Semantic Segmentation.

    Args:
        gt_semantic_masks (list[torch.Tensor]): Ground truth semantic labels.
        gt_instance_masks (list[torch.Tensor]): Ground truth instance labels.
        pred_instance_masks (list[torch.Tensor]): Predicted semantic labels.
        pred_semantic_masks  (list[torch.Tensor]): predicted instance masks.
        label2cat (dict): Map from label to category name.
        ignore_index (int): Index that will be ignored in evaluation.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Returns:
        dict[str, float]: Dict of results.
    """
    assert len(pred_semantic_masks) == len(gt_semantic_masks)
    num_classes = len(label2cat)

    hist_list = []
    for i in range(len(gt_semantic_masks)):
        gt_seg = gt_semantic_masks[i].astype(np.int64)
        pred_seg = pred_semantic_masks[i].astype(np.int64)

        # filter out ignored points
        pred_seg[gt_seg == ignore_index] = -1
        gt_seg[gt_seg == ignore_index] = -1

        # calculate one instance result
        hist_list.append(fast_hist(pred_seg, gt_seg, num_classes))

    iou = per_class_iou(sum(hist_list))
    # if ignore_index is in iou, replace it with nan
    if ignore_index < len(iou):
        iou[ignore_index] = np.nan
    miou = np.nanmean(iou)
    acc = get_acc(sum(hist_list))
    acc_cls = get_acc_cls(sum(hist_list))

    header = ['classes']
    for i in range(len(label2cat)):
        header.append(label2cat[i])
    header.extend(['miou', 'acc', 'acc_cls'])

    ret_dict = dict()
    table_columns = [['results']]
    for i in range(len(label2cat)):
        ret_dict[label2cat[i]] = float(iou[i])
        table_columns.append([f'{iou[i]:.4f}'])
    ret_dict['miou'] = float(miou)
    ret_dict['acc'] = float(acc)
    ret_dict['acc_cls'] = float(acc_cls)

    table_columns.append([f'{miou:.4f}'])
    table_columns.append([f'{acc:.4f}'])
    table_columns.append([f'{acc_cls:.4f}'])

    table_data = [header]
    table_rows = list(zip(*table_columns))
    table_data += table_rows
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)

    # Instance Segmentation Evaluation
    instance_coverage_list = []
    instance_weighted_coverage_list = []
    precision_list = []
    recall_list = []

    for gt_inst_mask, pred_inst_mask in zip(gt_instance_masks, pred_instance_masks):

        gt_instances = np.unique(gt_inst_mask)
        pred_instances = np.unique(pred_inst_mask)

        # (michbaum) Not sure if we want to ignore the ignore index here, since we
        #            don't want to penalize the model in any way for those points, no
        #            matter what it does with it
        gt_instances = gt_instances[gt_instances != ignore_index] 
        pred_instances = pred_instances[pred_instances]

        if len(gt_instances) == 0 or len(pred_instances) == 0:
            continue

        # Coverage and Weighted Coverage
        instance_coverage = []
        instance_weighted_coverage = []
        for gt_inst in gt_instances:
            gt_mask = (gt_inst_mask == gt_inst)
            max_iou = 0
            for pred_inst in pred_instances:
                pred_mask = (pred_inst_mask == pred_inst)
                iou_value = instance_iou(pred_mask, gt_mask)
                max_iou = max(max_iou, iou_value)
            instance_coverage.append(max_iou)
            instance_weighted_coverage.append(max_iou * gt_mask.sum())

        instance_coverage_list.append(np.mean(instance_coverage))
        instance_weighted_coverage_list.append(
            np.sum(instance_weighted_coverage) / gt_inst_mask.size
        )

        # Precision and Recall
        tp, fp, fn = 0, 0, 0
        matched_pred_instances = set()
        for gt_inst in gt_instances:
            gt_mask = (gt_inst_mask == gt_inst)
            matched = False
            for pred_inst in pred_instances:
                if pred_inst in matched_pred_instances:
                    continue
                pred_mask = (pred_inst_mask == pred_inst)
                if instance_iou(pred_mask, gt_mask) > 0.5:
                    tp += 1
                    matched_pred_instances.add(pred_inst)
                    matched = True
                    break
            if not matched:
                fn += 1
        fp = len(pred_instances) - len(matched_pred_instances)

        precision_list.append(tp / (tp + fp) if tp + fp > 0 else 0)
        recall_list.append(tp / (tp + fn) if tp + fn > 0 else 0)

    mean_coverage = np.mean(instance_coverage_list)
    mean_weighted_coverage = np.mean(instance_weighted_coverage_list)
    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    ret_dict['mean_coverage'] = float(mean_coverage)
    ret_dict['mean_weighted_coverage'] = float(mean_weighted_coverage)
    ret_dict['precision'] = float(precision)
    ret_dict['recall'] = float(recall)
    ret_dict['f1'] = float(f1)

    instance_metrics_header = ['Instance Metrics', 'Value']
    instance_metrics_table = [
        instance_metrics_header,
        ['mean_coverage', f'{mean_coverage:.4f}'],
        ['mean_weighted_coverage', f'{mean_weighted_coverage:.4f}'],
        ['precision', f'{precision:.4f}'],
        ['recall', f'{recall:.4f}'],
        ['f1', f'{f1:.4f}']
    ]
    instance_metrics_table = AsciiTable(instance_metrics_table)
    print_log('\n' + instance_metrics_table.table, logger=logger)

    return ret_dict