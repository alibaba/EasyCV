# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks

from easycv.models.face.face_keypoint import FaceKeypoint
from easycv.toolkit.modelscope.metainfo import EasyCVModels as Models
from easycv.toolkit.modelscope.models.base import EasyCVBaseModel


@MODELS.register_module(
    group_key=Tasks.face_2d_keypoints, module_name=Models.face_2d_keypoints)
class Face2DKeypoints(EasyCVBaseModel, FaceKeypoint):

    def __init__(self, model_dir=None, *args, **kwargs):
        EasyCVBaseModel.__init__(self, model_dir, args, kwargs)
        FaceKeypoint.__init__(self, *args, **kwargs)
