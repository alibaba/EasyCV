# Copyright (c) Alibaba, Inc. and its affiliates.
from .bevformer_predictor import BEVFormerPredictor
from .classifier import TorchClassifier
from .detector import (DetectionPredictor, TorchFaceDetector,
                       TorchYoloXClassifierPredictor, TorchYoloXPredictor,
                       YoloXPredictor)
from .face_keypoints_predictor import FaceKeypointsPredictor
from .feature_extractor import (TorchFaceAttrExtractor,
                                TorchFaceFeatureExtractor,
                                TorchFeatureExtractor)
from .hand_keypoints_predictor import HandKeypointsPredictor
from .ocr import (OCRClsPredictor, OCRDetPredictor, OCRPredictor,
                  OCRRecPredictor)
from .pose_predictor import (TorchPoseTopDownPredictor,
                             TorchPoseTopDownPredictorWithDetector)
from .segmentation import Mask2formerPredictor, SegmentationPredictor
from .video_classifier import VideoClassificationPredictor
from .wholebody_keypoints_predictor import WholeBodyKeypointsPredictor
