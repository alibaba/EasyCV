# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks

from easycv.models.detection.detectors import Detection as _Detection
from easycv.toolkit.modelscope.metainfo import EasyCVModels as Models
from easycv.toolkit.modelscope.models.base import EasyCVBaseModel


@MODELS.register_module(
    group_key=Tasks.image_object_detection, module_name=Models.dino)
class DINO(EasyCVBaseModel, _Detection):

    def __init__(self, model_dir=None, *args, **kwargs):
        EasyCVBaseModel.__init__(self, model_dir, args, kwargs)
        _Detection.__init__(self, *args, **kwargs)
