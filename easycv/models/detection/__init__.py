# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

from easycv.models.detection.dab_detr import DABDETRHead, DABDetrTransformer
from easycv.models.detection.detection import Detection
from easycv.models.detection.detr import DETRHead, DetrTransformer
from easycv.models.detection.fcos import FCOSHead
from easycv.models.detection.necks import *

try:
    from easycv.models.detection.yolox.yolox import YOLOX
except Exception as e:
    logging.info(f'Exception: {e}')
    logging.info(
        'Import YOLOX failed! please check your CUDA & Pytorch Version')

try:
    from easycv.models.detection.yolox_edge.yolox_edge import YOLOX_EDGE
except Exception as e:
    logging.info(f'Exception: {e}')
    logging.info(
        'Import YOLOX_EDGE failed! please check your CUDA & Pytorch Version')
