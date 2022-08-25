import copy
import json
import logging
import os

import cv2
import numpy as np
import torch
import torch.utils.data as data

from easycv.datasets.face.pipelines.face_keypoint_transform import (
    FaceKeypointNorm, FaceKeypointRandomAugmentation, normal)
from easycv.datasets.registry import DATASETS
from easycv.datasets.shared.base import BaseDataset


@DATASETS.register_module()
class FaceKeypointDataset(BaseDataset):
    """
        dataset for face key points
    """

    def __init__(self, data_source, pipeline, profiling=False):
        super(FaceKeypointDataset, self).__init__(data_source, pipeline,
                                                  profiling)
        """
        Args:
            data_source: Data_source config dict
            pipeline: Pipeline config list
            profiling: If set True, will print pipeline time
        """

    def evaluate(self, outputs, evaluators, **kwargs):
        eval_result = {}
        for evaluator in evaluators:
            eval_result.update(
                evaluator.evaluate(
                    prediction_dict=outputs,
                    groundtruth_dict=self.data_source.db))

        return eval_result

    def __getitem__(self, idx):
        results = self.data_source[idx]
        return self.pipeline(results)
