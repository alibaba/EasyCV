# Copyright (c) Alibaba, Inc. and its affiliates.
from .classifier import TorchClassifier
from .detector import (DetectionPredictor, TorchFaceDetector,
                       TorchYoloXClassifierPredictor, TorchYoloXPredictor)
from .face_keypoints_predictor import FaceKeypointsPredictor
from .feature_extractor import (TorchFaceAttrExtractor,
                                TorchFaceFeatureExtractor,
                                TorchFeatureExtractor)
from .hand_keypoints_predictor import HandKeypointsPredictor
from .pose_predictor import (TorchPoseTopDownPredictor,
                             TorchPoseTopDownPredictorWithDetector)
from .segmentation import Mask2formerPredictor, SegmentationPredictor
