# Copyright (c) Alibaba, Inc. and its affiliates.
import math

import numpy as np
import torch
from PIL import Image, ImageFile

from easycv.datasets.registry import PIPELINES
from easycv.file import io
from easycv.framework.errors import ValueError
from easycv.models.builder import build_model
from easycv.utils.misc import deprecated
from easycv.utils.mmlab_utils import (dynamic_adapt_for_mmlab,
                                      remove_adapt_for_mmlab)
from easycv.utils.registry import build_from_cfg
from .base import Predictor, PredictorV2
from .builder import PREDICTORS


@PREDICTORS.register_module()
class VideoClassificationPredictor(PredictorV2):
    """Predictor for classification.
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
    """

    def __init__(self,
                 model_path,
                 config_file=None,
                 batch_size=1,
                 device=None,
                 save_results=False,
                 save_path=None,
                 pipelines=None,
                 multi_class=False,
                 with_text=False,
                 label_map_path=None,
                 topk=1,
                 *args,
                 **kwargs):
        super(VideoClassificationPredictor, self).__init__(
            model_path,
            config_file=config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            pipelines=pipelines,
            *args,
            **kwargs)
        self.topk = topk
        self.multi_class = multi_class
        self.with_text = with_text

        if label_map_path is None:
            if 'CLASSES' in self.cfg:
                class_list = self.cfg.get('CLASSES', [])
            elif 'class_list' in self.cfg:
                class_list = self.cfg.get('class_list', [])
            elif 'num_classes' in self.cfg:
                class_list = list(range(self.cfg.get('num_classes', 0)))
                class_list = [str(i) for i in class_list]
            else:
                class_list = []
        else:
            with io.open(label_map_path, 'r') as f:
                class_list = f.readlines()
        self.label_map = [i.strip() for i in class_list]

    def _load_input(self, input):
        """Load image from file or numpy or PIL object.
        Args:
           {
                'filename': filename
            }
            or
           {
                'filename': filename,
                'text': text
            }
        """

        result = input
        if self.with_text and 'text' not in result:
            result['text'] = ''
        result['start_index'] = 0
        result['modality'] = 'RGB'

        return result

    def build_processor(self):
        """Build processor to process loaded input.
        If you need custom preprocessing ops, you need to reimplement it.
        """
        if self.pipelines is not None:
            pipelines = self.pipelines
        elif 'test_pipeline' in self.cfg:
            pipelines = self.cfg.get('test_pipeline', [])
        else:
            pipelines = self.cfg.get('val_pipeline', [])
        for idx, pipeline in enumerate(pipelines):
            if pipeline['type'] == 'Collect' and 'label' in pipeline['keys']:
                pipeline['keys'].remove('label')
            if pipeline['type'] == 'VideoToTensor' and 'label' in pipeline[
                    'keys']:
                pipeline['keys'].remove('label')
            pipelines[idx] = pipeline

        pipelines = [build_from_cfg(p, PIPELINES) for p in pipelines]

        from easycv.datasets.shared.pipelines.transforms import Compose
        processor = Compose(pipelines)
        return processor

    def _build_model(self):
        # Use mmdet model
        dynamic_adapt_for_mmlab(self.cfg)
        if 'vison_pretrained' in self.cfg.model:
            self.cfg.model.vison_pretrained = None
        if 'text_pretrained' in self.cfg.model:
            self.cfg.model.text_pretrained = None

        model = build_model(self.cfg.model)
        # remove adapt for mmdet to avoid conflict using mmdet models
        remove_adapt_for_mmlab(self.cfg)
        return model

    def postprocess(self, inputs, *args, **kwargs):
        """Return top-k results."""
        output_prob = inputs['prob'].data.cpu()
        topk_class = torch.topk(output_prob, self.topk).indices.numpy()
        output_prob = output_prob.numpy()
        batch_results = []
        batch_size = output_prob.shape[0]
        for i in range(batch_size):
            result = {'class': np.squeeze(topk_class[i]).tolist()}
            if isinstance(result['class'], int):
                result['class'] = [result['class']]

            if len(self.label_map) > 0:
                result['class_name'] = [
                    self.label_map[i] for i in result['class']
                ]
                result['class_probs'] = {}
                for l_idx, l_name in enumerate(self.label_map):
                    result['class_probs'][l_name] = output_prob[i][l_idx]

            batch_results.append(result)
        return batch_results
