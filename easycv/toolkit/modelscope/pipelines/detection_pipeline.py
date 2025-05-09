# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any

from modelscope.outputs import OutputKeys
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.cv.image_utils import \
    show_image_object_detection_auto_result

from easycv.toolkit.modelscope.metainfo import EasyCVPipelines as Pipelines
from .base import EasyCVPipeline


@PIPELINES.register_module(
    Tasks.image_object_detection, module_name=Pipelines.easycv_detection)
@PIPELINES.register_module(
    Tasks.image_object_detection,
    module_name=Pipelines.image_object_detection_auto)
@PIPELINES.register_module(
    Tasks.domain_specific_object_detection,
    module_name=Pipelines.hand_detection)
class EasyCVDetectionPipeline(EasyCVPipeline):
    """Pipeline for easycv detection task."""

    def __init__(self,
                 model: str,
                 model_file_pattern=ModelFile.TORCH_MODEL_FILE,
                 *args,
                 **kwargs):
        """
            model (str): model id on modelscope hub or local model path.
            model_file_pattern (str): model file pattern.
        """

        super(EasyCVDetectionPipeline, self).__init__(
            model=model,
            model_file_pattern=model_file_pattern,
            *args,
            **kwargs)

    def show_result(self, img_path, result, save_path=None):
        show_image_object_detection_auto_result(img_path, result, save_path)

    def __call__(self, inputs) -> Any:
        outputs = self.predict_op(inputs)

        scores = []
        labels = []
        boxes = []
        for output in outputs:
            scores_list = output['detection_scores'] if output[
                'detection_scores'] is not None else []
            classes_list = output['detection_classes'] if output[
                'detection_classes'] is not None else []
            boxes_list = output['detection_boxes'] if output[
                'detection_boxes'] is not None else []

            for score, label, box in zip(scores_list, classes_list,
                                         boxes_list):
                scores.append(score)
                labels.append(self.cfg.CLASSES[label])
                boxes.append([b for b in box])

        results = [{
            OutputKeys.SCORES: scores,
            OutputKeys.LABELS: labels,
            OutputKeys.BOXES: boxes
        } for output in outputs]

        if self._is_single_inputs(inputs):
            results = results[0]

        return results
