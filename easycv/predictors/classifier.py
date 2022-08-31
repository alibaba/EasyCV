# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import math
import os

import numpy as np
import torch
import torchvision
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from torch.hub import load_state_dict_from_url
from torchvision.transforms import Compose

from easycv.apis.export import reparameterize_models
from easycv.core.visualization import imshow_bboxes
from easycv.datasets.registry import PIPELINES
from easycv.datasets.utils import replace_ImageToTensor
from easycv.file import io
from easycv.file.utils import is_url_path, url_path_exists
from easycv.models import build_model
from easycv.models.detection.utils import postprocess
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.constant import CACHE_DIR
from easycv.utils.logger import get_root_logger
from easycv.utils.mmlab_utils import dynamic_adapt_for_mmlab
from easycv.utils.registry import build_from_cfg
from .base import Predictor
from .builder import PREDICTORS

try:
    from easy_vision.python.inference.predictor import PredictorInterface
except:
    from .interface import PredictorInterface


@PREDICTORS.register_module()
class TorchClassifier(PredictorInterface):

    def __init__(self,
                 model_path,
                 model_config=None,
                 topk=1,
                 label_map_path=None):
        """
    init model

    Args:
      model_path: model file path
      model_config: config string for model to init, in json format
    """
        self.predictor = Predictor(model_path)
        if 'class_list' not in self.predictor.cfg and label_map_path is None:
            raise Exception(
                "label_map_path need to be set, when ckpt doesn't contain class_list"
            )

        if label_map_path is None:
            class_list = self.predictor.cfg.get('class_list', [])
            self.label_map = [i.strip() for i in class_list]
        else:
            class_list = open(label_map_path).readlines()
            self.label_map = [i.strip() for i in class_list]
        self.output_name = ['prob', 'class']
        self.topk = topk if topk < len(class_list) else len(class_list)

    def get_output_type(self):
        """
    in this function user should return a type dict, which indicates
    which type of data should the output of predictor be converted to
    * type json, data will be serialized to json str

    * type image, data will be converted to encode image binary and write to oss file,
      whose name is output_dir/${key}/${input_filename}_${idx}.jpg, where input_filename
      is the base filename extracted from url, key corresponds to the key in the dict of output_type,
      if the type of data indexed by key is a list, idx is the index of element in list, otherwhile ${idx} will be empty

    * type video, data will be converted to encode video binary and write to oss file,

    :: return  {
      'image': 'image',
      'feature': 'json'
    }

    indicating that the image data in the output dict will be save to image
    file and feature in output dict will be converted to json

    """
        return {}

    def batch(self, image_tensor_list):
        return torch.stack(image_tensor_list)

    def predict(self, input_data_list, batch_size=-1):
        """
    using session run predict a number of samples using batch_size

    Args:
      input_data_list:  a list of numpy array, each array is a sample to be predicted
      batch_size: batch_size passed by the caller, you can also ignore this param and
        use a fixed number if you do not want to adjust batch_size in runtime
    Return:
      result: a list of dict, each dict is the prediction result of one sample
        eg, {"output1": value1, "output2": value2}, the value type can be
        python int str float, and numpy array
    """
        num_image = len(input_data_list)
        assert len(
            input_data_list) > 0, 'input images should not be an empty list'
        if batch_size > 0:
            num_batches = int(math.ceil(float(num_image) / batch_size))
            image_list = input_data_list
        else:
            num_batches = 1
            batch_size = len(input_data_list)
            image_list = input_data_list

        outputs_list = []
        for batch_idx in range(num_batches):
            batch_image_list = image_list[batch_idx * batch_size:min(
                (batch_idx + 1) * batch_size, len(image_list))]
            image_tensor_list = self.predictor.preprocess(batch_image_list)
            input_data = self.batch(image_tensor_list)
            output_prob = self.predictor.predict_batch(
                input_data, mode='test')['prob'].data.cpu()

            topk_prob = torch.topk(output_prob, self.topk).values.numpy()
            topk_class = torch.topk(output_prob, self.topk).indices.numpy()
            output_prob = output_prob.numpy()

            for idx in range(len(image_tensor_list)):
                single_result = {}
                single_result['class'] = np.squeeze(topk_class[idx]).tolist()
                if isinstance(single_result['class'], int):
                    single_result['class'] = [single_result['class']]
                single_result['class_name'] = [
                    self.label_map[i] for i in single_result['class']
                ]
                single_result['class_probs'] = {}
                for ldx, i in enumerate(self.label_map):
                    single_result['class_probs'][i] = output_prob[idx][ldx]

                outputs_list.append(single_result)

        return outputs_list


@PREDICTORS.register_module()
class ClsPredictor(PredictorInterface):
    """Inference image(s) with the detector.
    Args:
        model_path (str): checkpoint model and export model are shared.
        config_path (str): If config_path is specified, both checkpoint model and export model can be used; if config_path=None, the export model is used by default.
    """

    def __init__(self, model_path, config_path=None):

        self.model_path = model_path

        if config_path is not None:
            self.cfg = mmcv_config_fromfile(config_path)
        else:
            logger = get_root_logger()
            logger.warning('please use export model!')
            if is_url_path(self.model_path) and url_path_exists(
                    self.model_path):
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

    def predict(self, imgs):
        """
        Args:
            imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
            Either image files or loaded images.
        Returns:
            If imgs is a list or tuple, the same length list type results
            will be returned, otherwise return the detection results directly.
        """

        if isinstance(imgs, (list, tuple)):
            is_batch = True
        else:
            imgs = [imgs]
            is_batch = False

        cfg = self.cfg
        device = next(self.model.parameters()).device  # model device

        if isinstance(imgs[0], np.ndarray):
            cfg = cfg.copy()
            # set loading pipeline type
            cfg.data.val.pipeline.insert(0, dict(type='LoadImageFromWebcam'))
        else:
            cfg = cfg.copy()
            # set loading pipeline type
            cfg.data.val.pipeline.insert(
                0,
                dict(
                    type='LoadImageFromFile',
                    file_client_args=dict(
                        backend=('http' if imgs[0].startswith('http'
                                                              ) else 'disk'))))

        cfg.data.val.pipeline.insert(1, dict(type='ToPILImage'))
        cnt = 0
        for trans in cfg.data.val.pipeline:
            if trans['type'] == 'Collect':
                if 'gt_labels' in trans['keys']:
                    cfg.data.val.pipeline[cnt]['keys'].remove('gt_labels')
            cnt += 1
        cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)

        transforms = []
        for transform in cfg.data.val.pipeline:
            if 'img_scale' in transform:
                transform['img_scale'] = tuple(transform['img_scale'])
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                transforms.append(transform)
            elif callable(transform):
                transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')
        test_pipeline = Compose(transforms)

        datas = []
        for img in imgs:
            # prepare data
            if isinstance(img, np.ndarray):
                # directly add img
                data = dict(img=img)
            else:
                # add information into dict
                data = dict(img_info=dict(filename=img), img_prefix=None)
            # build the data pipeline
            data = test_pipeline(data)
            datas.append(data)

        data = collate(datas, samples_per_gpu=len(imgs))
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            for m in self.model.modules():
                assert not isinstance(
                    m, RoIPool
                ), 'CPU inference with RoIPool is not supported currently.'

        # forward the model
        with torch.no_grad():
            results = self.model(mode='test', img=data['img'][0].unsqueeze(0))

        return results
