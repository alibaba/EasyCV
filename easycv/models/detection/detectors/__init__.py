# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

from easycv.models.detection.detectors.dab_detr import (DABDETRHead,
                                                        DABDetrTransformer)
from easycv.models.detection.detectors.detection import Detection
from easycv.models.detection.detectors.detr import DETRHead, DetrTransformer
from easycv.models.detection.detectors.fcos import FCOSHead

try:
    from easycv.models.detection.detectors.yolox.yolox import YOLOX
except Exception as e:
    logging.info(f'Exception: {e}')
    logging.info(
        'Import YOLOX failed! please check your CUDA & Pytorch Version')

try:
    from easycv.models.detection.detectors.yolox_edge.yolox_edge import YOLOX_EDGE
except Exception as e:
    logging.info(f'Exception: {e}')
    logging.info(
        'Import YOLOX_EDGE failed! please check your CUDA & Pytorch Version')
