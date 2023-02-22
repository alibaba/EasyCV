# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os

import numpy as np
import torch
from PIL import Image, ImageFile

from easycv.file import io
from easycv.framework.errors import ValueError
from easycv.utils.misc import deprecated
from .base import InputProcessor, OutputProcessor, Predictor, PredictorV2
from .builder import PREDICTORS
from .classifier import ClassificationPredictor


@PREDICTORS.register_module()
class ReIDPredictor(ClassificationPredictor):
    """Predictor for reid.
    Args:
        model_path (str): Path of model path.
        config_file (Optinal[str]): config file path for model and processor to init. Defaults to None.
        batch_size (int): batch size for forward.
        device (str): Support 'cuda' or 'cpu', if is None, detect device automatically.
        save_results (bool): Whether to save predict results.
        save_path (str): File path for saving results, only valid when `save_results` is True.
        pipelines (list[dict]): Data pipeline configs.
        topk (int): Return top-k results. Default: 1.
        pil_input (bool): Whether use PIL image. If processor need PIL input, set true, default false.
        label_map_path (str): File path of saving labels list.
        input_processor_threads (int): Number of processes to process inputs.
        mode (str): The image mode into the model.
    """

    def __init__(self,
                 model_path,
                 config_file=None,
                 batch_size=1,
                 device=None,
                 save_results=False,
                 save_path=None,
                 pipelines=None,
                 topk=1,
                 pil_input=True,
                 label_map_path=None,
                 input_processor_threads=8,
                 mode='BGR',
                 *args,
                 **kwargs):

        if pil_input:
            mode = 'RGB'
        super(ReIDPredictor, self).__init__(
            model_path,
            config_file=config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            pipelines=pipelines,
            input_processor_threads=input_processor_threads,
            mode=mode,
            topk=topk,
            pil_input=pil_input,
            label_map_path=label_map_path,
            *args,
            **kwargs)

    def model_forward(self, inputs):
        """Model forward.
        If you need refactor model forward, you need to reimplement it.
        """
        with torch.no_grad():
            outputs = self.model(**inputs, mode='extract')
        return outputs

    def get_id(self, img_path):
        camera_id = []
        labels = []
        for path in img_path:
            filename = os.path.basename(path)
            label = filename[0:4]
            camera = filename.split('c')[1]
            if label[0:2] == '-1':
                labels.append(-1)
            else:
                labels.append(int(label))
            camera_id.append(int(camera[0]))
        return camera_id, labels

    def file_name_walk(self, file_dir):
        image_path_list = []
        for root, dirs, files in os.walk(file_dir):
            for name in files:
                if name.endswith('.jpg'):
                    image_path_list.append(os.path.join(root, name))
        return image_path_list

    def __call__(self, inputs, keep_inputs=False):
        if not isinstance(inputs, list):
            inputs = self.file_name_walk(inputs)
            img_cam, img_label = self.get_id(inputs)
        else:
            img_cam, img_label = None, None

        if self.input_processor is None:
            self.input_processor = self.get_input_processor()

        # TODO: fault tolerance
        if isinstance(inputs, (str, np.ndarray, ImageFile.ImageFile)):
            inputs = [inputs]

        results_list = []
        for i in range(0, len(inputs), self.batch_size):
            batch_inputs = inputs[i:min(len(inputs), i + self.batch_size)]
            batch_outputs = self.input_processor(batch_inputs)
            batch_outputs = self._to_device(batch_outputs)
            batch_outputs = self.model_forward(batch_outputs)
            results = batch_outputs['neck']
            if keep_inputs:
                for i in range(len(batch_inputs)):
                    results[i].update({'inputs': batch_inputs[i]})
            if isinstance(results, list):
                results_list.extend(results)
            else:
                results_list.append(results)

        image_feature = torch.cat(results_list, 0)
        image_feature_norm = torch.norm(
            image_feature, p=2, dim=1, keepdim=True)
        image_feature = image_feature.div(
            image_feature_norm.expand_as(image_feature))

        # TODO: support append to file
        if self.save_results:
            self.dump(image_feature, self.save_path)

        return {
            'img_feature': image_feature,
            'img_cam': img_cam,
            'img_label': img_label
        }
