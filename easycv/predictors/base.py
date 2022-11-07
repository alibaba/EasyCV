# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
import pickle

import cv2
import numpy as np
import torch
from mmcv.parallel import collate, scatter_kwargs
from PIL import Image, ImageFile
from torch.hub import load_state_dict_from_url
from torchvision.transforms import Compose

from easycv.datasets.registry import PIPELINES
from easycv.file import io
from easycv.file.utils import is_url_path
from easycv.framework.errors import ValueError
from easycv.models.builder import build_model
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.config_tools import Config, mmcv_config_fromfile
from easycv.utils.constant import CACHE_DIR
from easycv.utils.mmlab_utils import (dynamic_adapt_for_mmlab,
                                      remove_adapt_for_mmlab)
from easycv.utils.registry import build_from_cfg


class NumpyToPIL(object):

    def __call__(self, results):
        img = results['img']
        results['img'] = Image.fromarray(np.uint8(img)).convert('RGB')
        return results


class Predictor(object):

    def __init__(self, model_path, numpy_to_pil=True):
        self.model_path = model_path
        self.numpy_to_pil = numpy_to_pil
        assert io.exists(self.model_path), f'{self.model_path} does not exists'

        with io.open(self.model_path, 'rb') as infile:
            checkpoint = torch.load(infile, map_location='cpu')

        assert 'meta' in checkpoint and 'config' in checkpoint[
            'meta'], 'meta.config is missing from checkpoint'

        config_str = checkpoint['meta']['config']
        # get config
        basename = os.path.basename(self.model_path)
        fname, _ = os.path.splitext(basename)
        self.local_config_file = os.path.join(CACHE_DIR,
                                              f'{fname}_config.json')
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        with open(self.local_config_file, 'w') as ofile:
            ofile.write(config_str)
        self.cfg = mmcv_config_fromfile(self.local_config_file)

        # build model
        self.model = build_model(self.cfg.model)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        map_location = 'cpu' if self.device == 'cpu' else 'cuda'
        self.ckpt = load_checkpoint(
            self.model, self.model_path, map_location=map_location)

        self.model.to(self.device)
        self.model.eval()

        # build pipeline
        pipeline = [
            build_from_cfg(p, PIPELINES) for p in self.cfg.test_pipeline
        ]
        if self.numpy_to_pil:
            pipeline = [NumpyToPIL()] + pipeline
        self.pipeline = Compose(pipeline)

    def preprocess(self, image_list):
        # only perform transform to img
        output_imgs_list = []
        for img in image_list:
            tmp_input = {'img': img}
            tmp_results = self.pipeline(tmp_input)
            output_imgs_list.append(tmp_results['img'])

        return output_imgs_list

    def predict_batch(self, image_batch, **forward_kwargs):
        """ predict using batched data

    Args:
      image_batch(torch.Tensor): tensor with shape [N, 3, H, W]
      forward_kwargs: kwargs for additional parameters

    Return:
      output: the output of model.forward, list or tuple
    """
        with torch.no_grad():
            output = self.model.forward(
                image_batch.to(self.device), **forward_kwargs)
        return output


class PredictorV2(object):
    """Base predict pipeline.
        Args:
            model_path (str): Path of model path.
            config_file (Optinal[str]): config file path for model and processor to init. Defaults to None.
            batch_size (int): batch size for forward.
            device (str | torch.device): Support str('cuda' or 'cpu') or torch.device, if is None, detect device automatically.
            save_results (bool): Whether to save predict results.
            save_path (str): File path for saving results, only valid when `save_results` is True.
            pipelines (list[dict]): Data pipeline configs.
        """
    INPUT_IMAGE_MODE = 'BGR'  # the image mode into the model

    def __init__(self,
                 model_path,
                 config_file=None,
                 batch_size=1,
                 device=None,
                 save_results=False,
                 save_path=None,
                 pipelines=None,
                 *args,
                 **kwargs):
        self.model_path = model_path
        self.batch_size = batch_size
        self.save_results = save_results
        self.save_path = save_path
        self.config_file = config_file
        if self.save_results:
            assert self.save_path is not None
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if config_file is not None:
            if isinstance(config_file, str):
                self.cfg = mmcv_config_fromfile(config_file)
            else:
                self.cfg = config_file
        else:
            self.cfg = self._load_cfg_from_ckpt(self.model_path)

        if self.cfg is None:
            raise ValueError('Please provide "config_file"!')

        # avoid unnecessarily loading backbone weights from url
        if 'model' in self.cfg and 'pretrained' in self.cfg.model:
            self.cfg.model.pretrained = None

        self.model = self.prepare_model()
        self.pipelines = pipelines
        self.processor = self.build_processor()
        self._load_op = None

    def _load_cfg_from_ckpt(self, model_path):
        if is_url_path(model_path):
            ckpt = load_state_dict_from_url(model_path)
        else:
            with io.open(model_path, 'rb') as infile:
                ckpt = torch.load(infile, map_location='cpu')

        cfg = None
        if 'meta' in ckpt and 'config' in ckpt['meta']:
            cfg = ckpt['meta']['config']
            if isinstance(cfg, dict):
                cfg = Config(cfg)
            elif isinstance(cfg, str):
                cfg = Config(json.loads(cfg))
        return cfg

    def prepare_model(self):
        """Build model from config file by default.
        If the model is not loaded from a configuration file, e.g. torch jit model, you need to reimplement it.
        """
        model = self._build_model()
        model.to(self.device)
        model.eval()
        load_checkpoint(model, self.model_path, map_location='cpu')
        return model

    def _build_model(self):
        # Use mmdet model
        dynamic_adapt_for_mmlab(self.cfg)
        model = build_model(self.cfg.model)
        # remove adapt for mmdet to avoid conflict using mmdet models
        remove_adapt_for_mmlab(self.cfg)
        return model

    def build_processor(self):
        """Build processor to process loaded input.
        If you need custom preprocessing ops, you need to reimplement it.
        """
        if self.pipelines is not None:
            pipelines = self.pipelines
        else:
            pipelines = self.cfg.get('test_pipeline', [])

        pipelines = [build_from_cfg(p, PIPELINES) for p in pipelines]

        from easycv.datasets.shared.pipelines.transforms import Compose
        processor = Compose(pipelines)
        return processor

    def _load_input(self, input):
        """Load image from file or numpy or PIL object.
        Args:
            input: File path or numpy or PIL object.
        Returns:
           {
                'filename': filename,
                'img': img,
                'img_shape': img_shape,
                'img_fields': ['img']
            }
        """
        if self._load_op is None:
            load_cfg = dict(type='LoadImage', mode=self.INPUT_IMAGE_MODE)
            self._load_op = build_from_cfg(load_cfg, PIPELINES)

        if not isinstance(input, str):
            if isinstance(input, np.ndarray):
                # Only support RGB mode if input is np.ndarray.
                input = cv2.cvtColor(input, cv2.COLOR_RGB2BGR)
            sample = self._load_op({'img': input})
        else:
            sample = self._load_op({'filename': input})

        return sample

    def preprocess_single(self, input):
        """Preprocess single input sample.
        If you need custom ops to load or process a single input sample, you need to reimplement it.
        """
        input = self._load_input(input)
        return self.processor(input)

    def preprocess(self, inputs, *args, **kwargs):
        """Process all inputs list. And collate to batch and put to target device.
        If you need custom ops to load or process a batch samples, you need to reimplement it.
        """
        batch_outputs = []
        for i in inputs:
            batch_outputs.append(self.preprocess_single(i, *args, **kwargs))

        batch_outputs = self._collate_fn(batch_outputs)
        batch_outputs = self._to_device(batch_outputs)

        return batch_outputs

    def forward(self, inputs):
        """Model forward.
        If you need refactor model forward, you need to reimplement it.
        """
        with torch.no_grad():
            outputs = self.model(**inputs, mode='test')
        return outputs

    def _get_batch_size(self, inputs):
        for k, batch_v in inputs.items():
            if isinstance(batch_v, dict):
                batch_size = self._get_batch_size(batch_v)
            elif batch_v is not None:
                batch_size = len(batch_v)
                break
            else:
                batch_size = 1

        return batch_size

    def _extract_ith_result(self, inputs, i, out_i):
        for k, batch_v in inputs.items():
            if isinstance(batch_v, dict):
                out_i[k] = {}
                self._extract_ith_result(batch_v, i, out_i[k])
            elif batch_v is not None:
                out_i[k] = batch_v[i]
            else:
                out_i[k] = None
        return out_i

    def postprocess(self, inputs, *args, **kwargs):
        """Process model batch outputs.
        The "inputs" should be dict format as follows:
            {
                "key1": torch.Tensor or list, the first dimension should be batch_size,
                "key2": torch.Tensor or list, the first dimension should be batch_size,
                ...
            }
        """
        outputs = []
        batch_size = self._get_batch_size(inputs)
        for i in range(batch_size):
            out_i = self._extract_ith_result(inputs, i, {})
            out_i = self.postprocess_single(out_i, *args, **kwargs)
            outputs.append(out_i)
        return outputs

    def postprocess_single(self, inputs, *args, **kwargs):
        """Process outputs of single sample.
        If you need add some processing ops, you need to reimplement it.
        """
        return inputs

    def _collate_fn(self, inputs):
        """Prepare the input just before the forward function.
        Puts each data field into a tensor with outer dimension batch size
        """
        return collate(inputs, samples_per_gpu=self.batch_size)

    def _to_device(self, inputs):
        target_gpus = [-1] if str(
            self.device) == 'cpu' else [torch.cuda.current_device()]
        _, kwargs = scatter_kwargs(None, inputs, target_gpus=target_gpus)
        return kwargs[0]

    @staticmethod
    def dump(obj, save_path, mode='wb'):
        with open(save_path, mode) as f:
            f.write(pickle.dumps(obj))

    def __call__(self, inputs, keep_inputs=False):
        # TODO: fault tolerance

        if isinstance(inputs, (str, np.ndarray, ImageFile.ImageFile)):
            inputs = [inputs]

        results_list = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:min(len(inputs), i + self.batch_size)]
            batch_outputs = self.preprocess(batch)
            batch_outputs = self.forward(batch_outputs)
            results = self.postprocess(batch_outputs)
            # assert len(results) == len(
            #     batch), f'Mismatch size {len(results)} != {len(batch)}'
            if keep_inputs:
                for i in range(len(batch)):
                    results[i].update({'inputs': batch[i]})
            # if dump, the outputs will not added to the return value to prevent taking up too much memory
            if self.save_results:
                self.dump(results, self.save_path, mode='ab+')
            else:
                if isinstance(results, list):
                    results_list.extend(results)
                else:
                    results_list.append(results)
        return results_list
