# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import torch

from easycv.datasets.registry import PIPELINES
from easycv.file import io
from easycv.models.builder import build_model
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.mmlab_utils import (dynamic_adapt_for_mmlab,
                                      remove_adapt_for_mmlab)
from easycv.utils.registry import build_from_cfg
from .base import InputProcessor, OutputProcessor, PredictorV2
from .builder import PREDICTORS


class VideoClsInputProcessor(InputProcessor):

    def __init__(self,
                 cfg,
                 pipelines=None,
                 with_text=False,
                 batch_size=1,
                 threads=8,
                 mode='RGB'):
        self.with_text = with_text
        super().__init__(
            cfg,
            pipelines=pipelines,
            batch_size=batch_size,
            threads=threads,
            mode=mode)

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
        result['modality'] = self.mode

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


class VideoClsOutputProcessor(OutputProcessor):

    def __init__(self, label_map, topk=1):
        super().__init__()
        self.label_map = label_map
        self.topk = topk

    def __call__(self, inputs):
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
                 multi_class=False,
                 with_text=False,
                 label_map_path=None,
                 topk=1,
                 input_processor_threads=8,
                 mode='RGB',
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
            input_processor_threads=input_processor_threads,
            mode=mode,
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

    def get_input_processor(self):
        return VideoClsInputProcessor(
            self.cfg,
            pipelines=self.pipelines,
            with_text=self.with_text,
            batch_size=self.batch_size,
            threads=self.input_processor_threads,
            mode=self.mode)

    def get_output_processor(self):
        return VideoClsOutputProcessor(self.label_map, self.topk)


class STGCNInputProcessor(InputProcessor):

    def _load_input(self, input):
        """Prepare input sample.
        Args:
            input (dict): Input sample dict. e.g.
                {
                    'frame_dir': '',
                    'img_shape': (1080, 1920),
                    'original_shape': (1080, 1920),
                    'total_frames': 40,
                    'keypoint': (2, 40, 17, 2),  # shape = (num_person, num_frame, num_keypoints, 2)
                    'keypoint_score': (2, 40, 17),
                    'modality': 'Pose',
                    'start_index': 1
                }.
        """
        assert isinstance(input, dict)

        keypoint = input['keypoint']

        assert len(keypoint.shape) == 4
        assert keypoint.shape[-1] in [2, 3]

        if keypoint.shape[-1] == 3:
            if input.get('keypoint_score', None) is None:
                input['keypoint_score'] = keypoint[..., -1]

            keypoint = keypoint[..., :2]
            input['keypoint'] = keypoint

        return input


@PREDICTORS.register_module()
class STGCNPredictor(PredictorV2):
    """STGCN predict pipeline.
        Args:
            model_path (str): Path of model path.
            config_file (Optinal[str]): config file path for model and processor to init. Defaults to None.
            ori_image_size (Optinal[list|tuple]): Original image or video frame size (weight, height).
            batch_size (int): batch size for forward.
            label_map ((Optinal[list|tuple])): List or file of labels.
            device (str | torch.device): Support str('cuda' or 'cpu') or torch.device, if is None, detect device automatically.
            save_results (bool): Whether to save predict results.
            save_path (str): File path for saving results, only valid when `save_results` is True.
            pipelines (list[dict]): Data pipeline configs.
            input_processor_threads (int): Number of processes to process inputs.
            mode (str): The image mode into the model.
    """

    def __init__(self,
                 model_path,
                 config_file=None,
                 ori_image_size=None,
                 batch_size=1,
                 label_map=None,
                 topk=1,
                 device=None,
                 save_results=False,
                 save_path=None,
                 pipelines=None,
                 input_processor_threads=8,
                 mode='RGB',
                 model_type=None,
                 *args,
                 **kwargs):
        self.model_type = model_type
        if self.model_type is None:
            if model_path.endswith('jit'):
                assert config_file is not None
                self.model_type = 'jit'
            elif model_path.endswith('blade'):
                import torch_blade
                assert config_file is not None
                self.model_type = 'blade'
            else:
                self.model_type = 'raw'
        assert self.model_type in ['raw', 'jit', 'blade']

        super(STGCNPredictor, self).__init__(
            model_path,
            config_file=config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            pipelines=pipelines,
            input_processor_threads=input_processor_threads,
            mode=mode,
            *args,
            **kwargs)

        if ori_image_size is not None:
            w, h = ori_image_size
            for pipeline in self.cfg.test_pipeline:
                if pipeline['type'] == 'PoseNormalize':
                    pipeline['mean'] = (w // 2, h // 2, .5)
                    pipeline['max_value'] = (w, h, 1.)

        self.topk = topk
        if label_map is None:
            if 'CLASSES' in self.cfg:
                class_list = self.cfg.get('CLASSES', [])
            elif 'num_classes' in self.cfg:
                class_list = list(range(self.cfg.num_classes))
                class_list = [str(i) for i in class_list]
            else:
                class_list = []
        elif isinstance(label_map, str):
            with io.open(label_map, 'r') as f:
                class_list = f.readlines()
        elif isinstance(label_map, (tuple, list)):
            class_list = label_map

        self.label_map = [i.strip() for i in class_list]

    def _build_model(self):
        if self.model_type != 'raw':
            with io.open(self.model_path, 'rb') as infile:
                model = torch.jit.load(infile, self.device)
        else:
            model = super()._build_model()
        return model

    def prepare_model(self):
        """Build model from config file by default.
        If the model is not loaded from a configuration file, e.g. torch jit model, you need to reimplement it.
        """
        model = self._build_model()
        model.to(self.device)
        model.eval()
        if self.model_type == 'raw':
            load_checkpoint(model, self.model_path, map_location='cpu')
        return model

    def model_forward(self, inputs):
        if self.model_type == 'raw':
            return super().model_forward(inputs)
        else:
            with torch.no_grad():
                keypoint = inputs['keypoint'].to(self.device)
                result = self.model(keypoint)

        return result

    def get_input_processor(self):
        return STGCNInputProcessor(
            self.cfg,
            pipelines=self.pipelines,
            batch_size=self.batch_size,
            threads=self.input_processor_threads,
            mode=self.mode)

    def get_output_processor(self):
        return VideoClsOutputProcessor(self.label_map, self.topk)
