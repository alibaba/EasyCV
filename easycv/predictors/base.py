# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
from tabnanny import check

import numpy as np
import torch
from PIL import Image
from torch.hub import load_state_dict_from_url
from torchvision.transforms import Compose

from easycv.datasets.registry import PIPELINES
from easycv.file import io
from easycv.file.utils import is_url_path, url_path_exists
from easycv.models import build_model
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.constant import CACHE_DIR
from easycv.utils.mmlab_utils import dynamic_adapt_for_mmlab
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

        if is_url_path(self.model_path) and url_path_exists(self.model_path):
            checkpoint = load_state_dict_from_url(model_path)
        else:
            assert io.exists(
                self.model_path), f'{self.model_path} does not exists'

            with io.open(self.model_path, 'rb') as infile:
                checkpoint = torch.load(infile, map_location='cpu')

        assert 'meta' in checkpoint and 'config' in checkpoint[
            'meta'], 'meta.config is missing from checkpoint'

        config_str = checkpoint['meta']['config']
        if isinstance(config_str, dict):
            config_str = json.dumps(config_str)
        print(config_str)

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

        # dynamic adapt mmdet models
        dynamic_adapt_for_mmlab(self.cfg)

        # build model
        self.model = build_model(self.cfg.model)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        map_location = 'cpu' if self.device == 'cpu' else 'cuda'
        self.ckpt = load_checkpoint(
            self.model, self.model_path, map_location=map_location)

        self.model.to(self.device)
        self.model.eval()

        # build pipeline
        # transforms = []
        # for transform in self.cfg.test_pipeline:
        #     if isinstance(transform, dict):
        #         transform = build_from_cfg(transform, PIPELINES)
        #         transforms.append(transform)
        #     elif callable(transform):
        #         transforms.append(transform)
        #     else:
        #         raise TypeError('transform must be callable or a dict')
        # self.pipeline = Compose(transforms)

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
