# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks

from easycv.models.pose.top_down import TopDown
from easycv.toolkit.modelscope.metainfo import EasyCVModels as Models
from easycv.toolkit.modelscope.models.base import EasyCVBaseModel


@MODELS.register_module(
    group_key=Tasks.human_wholebody_keypoint,
    module_name=Models.human_wholebody_keypoint)
class HumanWholeBodyKeypoint(EasyCVBaseModel, TopDown):

    def __init__(self, model_dir=None, *args, **kwargs):
        EasyCVBaseModel.__init__(self, model_dir, args, kwargs)
        TopDown.__init__(self, *args, **kwargs)
