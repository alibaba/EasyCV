# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np

from easycv.datasets.registry import DATASETS
from easycv.datasets.shared.base import BaseDataset
from easycv.utils.bbox_util import batched_xyxy2cxcywh_with_shape


@DATASETS.register_module
class DetDataset(BaseDataset):
    """Dataset for Detection
    """

    def __init__(self, data_source, pipeline, profiling=False, classes=None):
        """
        Args:
            data_source: Data_source config dict
            pipeline: Pipeline config list
            profiling: If set True, will print pipeline time
            classes: A list of class names, used in evaluation for result and groundtruth visualization
        """
        self.classes = classes

        super(DetDataset, self).__init__(
            data_source, pipeline, profiling=profiling)
        self.img_num = self.data_source.get_length()

    def __getitem__(self, idx):
        data_dict = self.data_source.get_sample(idx)
        data_dict = self.pipeline(data_dict)
        return data_dict

    def evaluate(self, results, evaluators, logger=None):
        '''results: a dict of list of Tensors, list length equals to number of test images
        '''
        eval_result = dict()

        groundtruth_dict = {}
        groundtruth_dict['groundtruth_boxes'] = [
            batched_xyxy2cxcywh_with_shape(
                self.data_source.get_ann_info(idx)['bboxes'],
                results['img_metas'][idx]['ori_img_shape'])
            for idx in range(len(results['img_metas']))
        ]
        groundtruth_dict['groundtruth_classes'] = [
            self.data_source.get_ann_info(idx)['labels']
            for idx in range(len(results['img_metas']))
        ]
        groundtruth_dict['groundtruth_is_crowd'] = [
            self.data_source.get_ann_info(idx)['groundtruth_is_crowd']
            for idx in range(len(results['img_metas']))
        ]

        groundtruth_dict['groundtruth_instance_masks'] = [
            self.data_source.get_ann_info(idx).get('masks', None)
            for idx in range(len(results['img_metas']))
        ]

        for evaluator in evaluators:
            eval_result.update(evaluator.evaluate(results, groundtruth_dict))

        return eval_result
