# Copyright (c) Alibaba, Inc. and its affiliates.
import torch

from .base_evaluator import Evaluator
from .builder import EVALUATORS
from .metric_registry import METRICS


@EVALUATORS.register_module
class FaceKeypointEvaluator(Evaluator):

    def __init__(self, dataset_name=None, metric_names=['ave_nme']):
        super(FaceKeypointEvaluator, self).__init__(dataset_name, metric_names)
        self.metric = metric_names
        self.dataset_name = dataset_name

    def _evaluate_impl(self, prediction_dict, groundtruth_dict, **kwargs):
        """
        Args:
            prediction_dict: model forward output dict, ['point', 'pose']
            groundtruth_dict: groundtruth dict, ['target_point', 'target_point_mask', 'target_pose', 'target_pose_mask'] used for compute accuracy
            kwargs: other parameters
        """

        def evaluate(predicts, gts, **kwargs):
            from easycv.models.utils.face_keypoint_utils import get_keypoint_accuracy, get_pose_accuracy
            ave_pose_acc = 0
            ave_nme = 0
            idx = 0

            for (predict_point, predict_pose,
                 gt) in zip(predicts['point'], predicts['pose'], gts):
                target_point = gt['target_point']
                target_point_mask = gt['target_point_mask']
                target_pose = gt['target_pose']
                target_pose_mask = gt['target_pose_mask']

                target_point = target_point * target_point_mask
                target_pose = target_pose * target_pose_mask

                keypoint_accuracy = get_keypoint_accuracy(
                    predict_point, target_point)
                pose_accuracy = get_pose_accuracy(predict_pose, target_pose)

                ave_pose_acc += pose_accuracy['pose_acc']
                ave_nme += keypoint_accuracy['nme']
                idx += 1

            eval_result = {}
            idx += 0.000001
            eval_result['ave_pose_acc'] = ave_pose_acc / idx
            eval_result['ave_nme'] = ave_nme / idx

            return eval_result

        return evaluate(prediction_dict, groundtruth_dict)


METRICS.register_default_best_metric(FaceKeypointEvaluator, 'ave_nme', 'min')
