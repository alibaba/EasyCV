# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any

from modelscope.outputs import OutputKeys
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

from easycv.toolkit.modelscope.metainfo import EasyCVPipelines as Pipelines
from .base import EasyCVPipeline

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_segmentation,
    module_name=Pipelines.image_panoptic_segmentation_easycv)
class ImagePanopticSegmentationEasyCVPipeline(EasyCVPipeline):
    """Pipeline built upon easycv for image segmentation."""

    def __init__(self, model: str, model_file_pattern='*.pt', *args, **kwargs):
        """
            model (str): model id on modelscope hub or local model path.
            model_file_pattern (str): model file pattern.
        """
        super(ImagePanopticSegmentationEasyCVPipeline, self).__init__(
            model=model,
            model_file_pattern=model_file_pattern,
            *args,
            **kwargs)

    def __call__(self, inputs) -> Any:
        outputs = self.predict_op(inputs)
        easycv_results = outputs[0]

        results = {
            OutputKeys.MASKS:
            easycv_results[OutputKeys.MASKS],
            OutputKeys.LABELS:
            easycv_results[OutputKeys.LABELS],
            OutputKeys.SCORES:
            [0.999 for _ in range(len(easycv_results[OutputKeys.LABELS]))]
        }

        return results
