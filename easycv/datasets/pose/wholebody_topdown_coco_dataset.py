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
                'WholeBodyCocoTopDownDataset only support one `WholeBodyKeyPointEvaluator` now, '
                'but get %s' % evaluators)

        evaluator_args = {
            'num_joints': self.data_source.ann_info['num_joints'],
            'sigmas': self.data_source.sigmas,
            'class2id': self.data_source._class_to_ind
        }
        eval_result = {}
        for evaluator in evaluators:
            eval_result.update(
                evaluator.evaluate(
                    prediction_dict=outputs,
                    groundtruth_dict=self.data_source.coco.dataset,
                    **evaluator_args))

        return eval_result
