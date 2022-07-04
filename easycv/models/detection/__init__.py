# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

from .dab_detr import DABDETRHead, DABDetrTransformer
from .det_builder import DETRTRANSFORMER, build_detr_transformer
from .detection import Detection
from .detr import DETRHead, DetrTransformer
from .vitdet import SFP

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
