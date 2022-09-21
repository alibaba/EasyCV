# Copyright (c) Alibaba, Inc. and its affiliates.
from .cross_entropy_loss import CrossEntropyLoss
from .det_db_loss import DBLoss
from .face_keypoint_loss import FacePoseLoss, WingLossWithPose
from .focal_loss import FocalLoss, VarifocalLoss
from .iou_loss import GIoULoss, IoULoss, YOLOX_IOULoss
from .mse_loss import JointsMSELoss
from .ocr_rec_multi_loss import MultiLoss
from .pytorch_metric_learning import *
from .set_criterion import (CDNCriterion, DNCriterion, HungarianMatcher,
                            SetCriterion)
