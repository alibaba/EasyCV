# Copyright (c) Alibaba, Inc. and its affiliates.
# isort:skip_file
from easycv.utils.import_utils import check_numpy
check_numpy()
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
from .wholebody_keypoint_eval import WholeBodyKeyPointEvaluator
try:
    from .nuscenes_eval import NuScenesEvaluator
except ImportError as e:
    pass
