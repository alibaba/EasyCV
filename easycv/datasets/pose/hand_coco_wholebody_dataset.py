# Copyright (c) OpenMMLab. All rights reserved.
# Adapt from
# https://github.com/open-mmlab/mmpose/blob/master/mmpose/datasets/datasets/hand/hand_coco_wholebody_dataset.py

from easycv.core.evaluation.keypoint_eval import KeyPointEvaluator
from easycv.datasets.pose.data_sources.coco import PoseTopDownSource
from easycv.datasets.registry import DATASETS
from easycv.datasets.shared.base import BaseDataset
from easycv.framework.errors import ValueError


@DATASETS.register_module()
class HandCocoWholeBodyDataset(BaseDataset):
    """CocoWholeBodyDataset for top-down hand pose estimation.

    Args:
        data_source: Data_source config dict
        pipeline: Pipeline config list
        profiling: If set True, will print pipeline time
    """

    def __init__(self, data_source, pipeline, profiling=False):
        super(HandCocoWholeBodyDataset, self).__init__(data_source, pipeline,
                                                       profiling)

        if not isinstance(self.data_source, PoseTopDownSource):
            raise ValueError('Only support `PoseTopDownSource`, but get %s' %
                             self.data_source)

    def evaluate(self, outputs, evaluators, **kwargs):
        if len(evaluators) > 1 or not isinstance(evaluators[0],
                                                 KeyPointEvaluator):
            raise ValueError(
                'HandCocoWholeBodyDataset only support one `KeyPointEvaluator` now, '
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

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts

    def __getitem__(self, idx):
        """Get the sample given index."""
        results = self.data_source[idx]
        return self.pipeline(results)
