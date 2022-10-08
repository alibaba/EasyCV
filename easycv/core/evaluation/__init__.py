# Copyright (c) Alibaba, Inc. and its affiliates.
from .auc_eval import AucEvaluator
from .base_evaluator import Evaluator
from .classification_eval import ClsEvaluator
from .coco_evaluation import CocoDetectionEvaluator, CoCoPoseTopDownEvaluator
from .face_eval import FaceKeypointEvaluator
from .faceid_pair_eval import FaceIDPairEvaluator
from .keypoint_eval import KeyPointEvaluator
from .mse_eval import MSEEvaluator
from .ocr_eval import OCRDetEvaluator, OCRRecEvaluator
from .retrival_topk_eval import RetrivalTopKEvaluator
from .segmentation_eval import SegmentationEvaluator
from .top_down_eval import (keypoint_auc, keypoint_epe, keypoint_nme,
                            keypoint_pck_accuracy, keypoints_from_heatmaps,
                            pose_pck_accuracy)
