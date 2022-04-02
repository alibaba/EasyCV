# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.core.evaluation.coco_evaluation import CoCoPoseTopDownEvaluator
from easycv.datasets.pose.data_sources.coco import PoseTopDownSource
from easycv.datasets.registry import DATASETS
from easycv.datasets.shared.base import BaseDataset


@DATASETS.register_module()
class PoseTopDownDataset(BaseDataset):
    """PoseTopDownDataset dataset for top-down pose estimation.
    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Args:
        data_source: Data_source config dict
        pipeline: Pipeline config list
        profiling: If set True, will print pipeline time
    """

    def __init__(self, data_source, pipeline, profiling=False):
        super(PoseTopDownDataset, self).__init__(data_source, pipeline,
                                                 profiling)

        if not isinstance(self.data_source, PoseTopDownSource):
            raise ValueError('Only support `PoseTopDownSource`, but get %s' %
                             self.data_source)

    def evaluate(self, outputs, evaluators, **kwargs):
        if len(evaluators) > 1 or not isinstance(evaluators[0],
                                                 CoCoPoseTopDownEvaluator):
            raise ValueError(
                'PoseTopDownDataset only support one `CoCoPoseTopDownEvaluator` now, '
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

    def __getitem__(self, idx):
        """Get the sample given index."""
        results = self.data_source.get_sample(idx)

        return self.pipeline(results)
