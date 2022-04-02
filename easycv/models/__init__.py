# Copyright (c) Alibaba, Inc. and its affiliates.
from .backbones import *  # noqa: F401,F403
from .builder import build_backbone, build_head, build_loss, build_model
from .classification import *
from .heads import *
from .loss import *
from .pose import TopDown
from .registry import BACKBONES, HEADS, LOSSES, MODELS, NECKS
from .selfsup import *

try:
    from .detection.yolox.yolox import YOLOX
except:
    import logging
    logging.warning(
        'Import YOLOX failed! Please check if mmcv and CUDA & Pytorch match.'
        'You may try: `pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html`.'
        'e.g.: `pip install mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html`'
    )
try:
    from .detection.yolox_edge.yolox_edge import YOLOX_EDGE
except:
    import logging
    logging.warning(
        'Import YOLOX EDGE model failed! Please check if mmcv and CUDA & Pytorch match.'
    )
