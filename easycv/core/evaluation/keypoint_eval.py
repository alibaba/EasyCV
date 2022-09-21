# Copyright (c) OpenMMLab. All rights reserved.
# Adapt from
# https://github.com/open-mmlab/mmpose/blob/master/mmpose/datasets/datasets/base/kpt_2d_sview_rgb_img_top_down_dataset.py
import numpy as np

from easycv.framework.errors import KeyError
from .base_evaluator import Evaluator
from .builder import EVALUATORS
from .metric_registry import METRICS
from .top_down_eval import (keypoint_auc, keypoint_epe, keypoint_nme,
                            keypoint_pck_accuracy)


@EVALUATORS.register_module
class KeyPointEvaluator(Evaluator):
    """ KeyPoint evaluator.
    """

    def __init__(self,
                 dataset_name=None,
                 metric_names=['PCK', 'PCKh', 'AUC', 'EPE', 'NME'],
                 pck_thr=0.2,
                 pckh_thr=0.7,
                 auc_nor=30):
        """

        Args:
            dataset_name: eval dataset name
            metric_names: eval metrics name
            pck_thr (float): PCK threshold, default as 0.2.
            pckh_thr (float): PCKh threshold, default as 0.7.
            auc_nor (float): AUC normalization factor, default as 30 pixel.
        """
        super(KeyPointEvaluator, self).__init__(dataset_name, metric_names)
        self._pck_thr = pck_thr
        self._pckh_thr = pckh_thr
        self._auc_nor = auc_nor
        self.dataset_name = dataset_name
        allowed_metrics = ['PCK', 'PCKh', 'AUC', 'EPE', 'NME']
        for metric in metric_names:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

    def _evaluate_impl(self, preds, coco_db, **kwargs):
        ''' keypoint evaluation code which will be run after
        all test batched data are predicted

        Args:
            preds: dict with key ``keypoints`` whose shape is Nx3
            coco_db: the db of wholebody coco datasource, sorted by 'bbox_id'

        Return:
            a dict,  each key is metric_name, value is metric value
        '''
        assert len(preds) == len(coco_db)
        eval_res = {}

        outputs = []
        gts = []
        masks = []
        box_sizes = []
        threshold_bbox = []
        threshold_head_box = []

        for pred, item in zip(preds, coco_db):
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(item['joints_3d'])[:, :-1])
            masks.append((np.array(item['joints_3d_visible'])[:, 0]) > 0)
            if 'PCK' in self.metric_names:
                bbox = np.array(item['bbox'])
                bbox_thr = np.max(bbox[2:])
                threshold_bbox.append(np.array([bbox_thr, bbox_thr]))
            if 'PCKh' in self.metric_names:
                head_box_thr = item['head_size']
                threshold_head_box.append(
                    np.array([head_box_thr, head_box_thr]))
            box_sizes.append(item.get('box_size', 1))

        outputs = np.array(outputs)
        gts = np.array(gts)
        masks = np.array(masks)
        threshold_bbox = np.array(threshold_bbox)
        threshold_head_box = np.array(threshold_head_box)
        box_sizes = np.array(box_sizes).reshape([-1, 1])

        if 'PCK' in self.metric_names:
            _, pck, _ = keypoint_pck_accuracy(outputs, gts, masks,
                                              self._pck_thr, threshold_bbox)
            eval_res['PCK'] = pck

        if 'PCKh' in self.metric_names:
            _, pckh, _ = keypoint_pck_accuracy(outputs, gts, masks,
                                               self._pckh_thr,
                                               threshold_head_box)
            eval_res['PCKh'] = pckh

        if 'AUC' in self.metric_names:
            eval_res['AUC'] = keypoint_auc(outputs, gts, masks, self._auc_nor)

        if 'EPE' in self.metric_names:
            eval_res['EPE'] = keypoint_epe(outputs, gts, masks)

        if 'NME' in self.metric_names:
            normalize_factor = self._get_normalize_factor(
                gts=gts, box_sizes=box_sizes)
            eval_res['NME'] = keypoint_nme(outputs, gts, masks,
                                           normalize_factor)
        return eval_res

    def _get_normalize_factor(self, gts, *args, **kwargs):
        """Get the normalize factor. generally inter-ocular distance measured
        as the Euclidean distance between the outer corners of the eyes is
        used. This function should be overrode, to measure NME.

        Args:
            gts (np.ndarray[N, K, 2]): Groundtruth keypoint location.

        Returns:
            np.ndarray[N, 2]: normalized factor
        """
        return np.ones([gts.shape[0], 2], dtype=np.float32)


METRICS.register_default_best_metric(KeyPointEvaluator, 'PCK', 'max')
