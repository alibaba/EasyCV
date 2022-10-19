# Copyright (c) OpenMMLab. All rights reserved.
import os

import numpy as np

from easycv.core.evaluation.wholebody_keypoint_eval import \
    WholeBodyKeyPointEvaluator
from easycv.datasets.pose.data_sources.coco import PoseTopDownSource
from easycv.datasets.registry import DATASETS
from easycv.datasets.shared.raw import RawDataset


@DATASETS.register_module()
class WholeBodyCocoTopDownDataset(RawDataset):
    """CocoWholeBodyDataset dataset for top-down pose estimation.

    Args:
        data_source: Data_source config dict
        pipeline: Pipeline config list
        profiling: If set True, will print pipeline time
    """

    def __init__(self, data_source, pipeline, profiling=False):
        super(WholeBodyCocoTopDownDataset,
              self).__init__(data_source, pipeline, profiling)

    def evaluate(self, outputs, evaluators, **kwargs):
        if len(evaluators) > 1 or not isinstance(evaluators[0],
                                                 WholeBodyKeyPointEvaluator):
            raise ValueError(
                'HandCocoWholeBodyDataset only support one `WholeBodyKeyPointEvaluator` now, '
                'but get %s' % evaluators)
        evaluator = evaluators[0]

        image_ids = outputs['image_ids']
        preds = outputs['preds']
        boxes = outputs['boxes']
        bbox_ids = outputs['bbox_ids']

        kpts = []
        for i, image_id in enumerate(image_ids):
            kpts.append({
                'keypoints': preds[i],
                'center': boxes[i][0:2],
                'scale': boxes[i][2:4],
                'area': boxes[i][4],
                'score': boxes[i][5],
                'image_id': image_id,
                'bbox_id': bbox_ids[i]
            })
        kpts = self._sort_and_unique_bboxes(kpts)
        eval_res = evaluator.evaluate(kpts, self.data_source.db)
        return eval_res
