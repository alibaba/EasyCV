# Copyright (c) Alibaba, Inc. and its affiliates.
from .bevformer_predictor import BEVFormerPredictor
from .classifier import ClassificationPredictor, TorchClassifier
from .detector import (DetectionPredictor, TorchFaceDetector,
                       TorchYoloXClassifierPredictor, TorchYoloXPredictor,
                       YoloXPredictor)
from .face_keypoints_predictor import FaceKeypointsPredictor
from .feature_extractor import (TorchFaceAttrExtractor,
                                TorchFaceFeatureExtractor,
                                TorchFeatureExtractor)
from .hand_keypoints_predictor import HandKeypointsPredictor
from .mot_predictor import MOTPredictor
from .ocr import (OCRClsPredictor, OCRDetPredictor, OCRPredictor,
                  OCRRecPredictor)
from .pose_predictor import (PoseTopDownPredictor,
                             TorchPoseTopDownPredictorWithDetector)
from .reid_predictor import ReIDPredictor
from .segmentation import Mask2formerPredictor, SegmentationPredictor
from .video_classifier import VideoClassificationPredictor
from .wholebody_keypoints_predictor import WholeBodyKeypointsPredictor
