# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import pickle

import numpy as np
import torch
from mmcv.parallel import collate, scatter_kwargs
from PIL import Image
from torchvision.transforms import Compose

from easycv.datasets.registry import PIPELINES
from easycv.file import io
from easycv.models.builder import build_model
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.constant import CACHE_DIR
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
            device (str): Support 'cuda' or 'cpu', if is None, detect device automatically.
            save_results (bool): Whether to save predict results.
            save_path (str): File path for saving results, only valid when `save_results` is True.
        """

    def __init__(self,
                 model_path,
                 config_file=None,
                 batch_size=1,
                 device=None,
                 save_results=False,
                 save_path=None,
                 *args,
                 **kwargs):
        self.model_path = model_path
        self.batch_size = batch_size
        self.save_results = save_results
        self.save_path = save_path
        if self.save_results:
            assert self.save_path is not None
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.cfg = None
        if config_file is not None:
            if isinstance(config_file, str):
                self.cfg = mmcv_config_fromfile(config_file)
            else:
                self.cfg = config_file

        self.model = self.prepare_model()
        self.processor = self.build_processor()
        self._load_op = None

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
        if self.cfg is None:
            raise ValueError('Please provide "config_file"!')

        model = build_model(self.cfg.model)
        return model

    def build_processor(self):
        """Build processor to process loaded input.
        If you need custom preprocessing ops, you need to reimplement it.
        """
        if self.cfg is None:
            pipeline = []
        else:
            pipeline = [
                build_from_cfg(p, PIPELINES)
                for p in self.cfg.get('test_pipeline', [])
            ]

        from easycv.datasets.shared.pipelines.transforms import Compose
        processor = Compose(pipeline)
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
            load_cfg = dict(type='LoadImage', mode='rgb')
            self._load_op = build_from_cfg(load_cfg, PIPELINES)

        if not isinstance(input, str):
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

    def postprocess(self, inputs, *args, **kwargs):
        """Process model outputs.
        If you need add some processing ops to process model outputs, you need to reimplement it.
        """
        return inputs

    def _collate_fn(self, inputs):
        """Prepare the input just before the forward function.
        Puts each data field into a tensor with outer dimension batch size
        """
        return collate(inputs, samples_per_gpu=self.batch_size)

    def _to_device(self, inputs):
        target_gpus = [-1] if self.device == 'cpu' else [
            torch.cuda.current_device()
        ]
        _, kwargs = scatter_kwargs(None, inputs, target_gpus=target_gpus)
        return kwargs[0]

    @staticmethod
    def dump(obj, save_path, mode='wb'):
        with open(save_path, mode) as f:
            f.write(pickle.dumps(obj))

    def __call__(self, inputs, keep_inputs=False):
        # TODO: fault tolerance

        if isinstance(inputs, str):
            inputs = [inputs]

        results_list = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:max(len(inputs) - 1, i + self.batch_size)]
            batch_outputs = self.preprocess(batch)
            batch_outputs = self.forward(batch_outputs)
            results = self.postprocess(batch_outputs)
            if keep_inputs:
                results = {'inputs': batch, 'results': results}
            # if dump, the outputs will not added to the return value to prevent taking up too much memory
            if self.save_results:
                self.dump([results], self.save_path, mode='ab+')
            else:
                results_list.append(results)

        return results_list
