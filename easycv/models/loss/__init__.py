# Copyright (c) Alibaba, Inc. and its affiliates.
from .cross_entropy_loss import CrossEntropyLoss
from .det_db_loss import DBLoss
from .dice_loss import DiceLoss
from .face_keypoint_loss import FacePoseLoss, WingLossWithPose
from .focal_loss import FocalLoss, VarifocalLoss
from .iou_loss import GIoULoss, IoULoss, YOLOX_IOULoss
from .l1_loss import L1Loss, SmoothL1Loss
from .mse_loss import JointsMSELoss
from .ocr_rec_multi_loss import MultiLoss
from .pytorch_metric_learning import (AMSoftmaxLoss,
                                      CrossEntropyLossWithLabelSmooth,
                                      DistributeMSELoss, FocalLoss2d,
                                      ModelParallelAMSoftmaxLoss,
                                      ModelParallelSoftmaxLoss,
                                      SoftTargetCrossEntropy)
from .set_criterion import (CDNCriterion, DNCriterion, HungarianMatcher,
                            SetCriterion)

__all__ = [
    'CrossEntropyLoss', 'FacePoseLoss', 'WingLossWithPose', 'FocalLoss',
    'VarifocalLoss', 'GIoULoss', 'IoULoss', 'YOLOX_IOULoss', 'JointsMSELoss',
    'FocalLoss2d', 'DistributeMSELoss', 'CrossEntropyLossWithLabelSmooth',
    'AMSoftmaxLoss', 'ModelParallelSoftmaxLoss', 'ModelParallelAMSoftmaxLoss',
    'SoftTargetCrossEntropy', 'CDNCriterion', 'DNCriterion', 'DBLoss',
    'HungarianMatcher', 'SetCriterion', 'L1Loss', 'MultiLoss', 'SmoothL1Loss',
    'DiceLoss'
]
