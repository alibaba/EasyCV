# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

from .dab_detr import DABDETRHead, DABDetrTransformer
from .detr import DETR, DETRHead, DetrTransformer
from .fcos import FCOS, FPN, FCOSHead
from .vitdet import SFP, RPNHeadNorm

try:
    from .yolox.yolox import YOLOX
except Exception as e:
    logging.info(f'Exception: {e}')
    logging.info(
        'Import YOLOX failed! please check your CUDA & Pytorch Version')

try:
    from .yolox_edge.yolox_edge import YOLOX_EDGE
except Exception as e:
    logging.info(f'Exception: {e}')
    logging.info(
        'Import YOLOX_EDGE failed! please check your CUDA & Pytorch Version')
