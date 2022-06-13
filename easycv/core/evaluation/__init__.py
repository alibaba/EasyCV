# Copyright (c) Alibaba, Inc. and its affiliates.
from .auc_eval import AucEvaluator
from .base_evaluator import Evaluator
from .classification_eval import ClsEvaluator
from .coco_evaluation import CocoDetectionEvaluator, CoCoPoseTopDownEvaluator
from .faceid_pair_eval import FaceIDPairEvaluator
from .mse_eval import MSEEvaluator
from .retrival_topk_eval import RetrivalTopKEvaluator
from .segmentation_eval import SegmentationEvaluator
from .top_down_eval import (keypoint_pck_accuracy, keypoints_from_heatmaps,
                            pose_pck_accuracy)
