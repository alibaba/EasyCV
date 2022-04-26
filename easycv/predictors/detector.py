# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
from glob import glob

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose

from easycv.datasets.registry import PIPELINES
from easycv.file import io
from easycv.models import build_model
from easycv.utils.checkpoint import load_checkpoint
# from mmcv import Config
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.constant import CACHE_DIR
from easycv.utils.registry import build_from_cfg
from .builder import PREDICTORS
from .classifier import TorchClassifier

try:
    from easy_vision.python.inference.predictor import PredictorInterface
except Exception:
    from .interface import PredictorInterface

try:
    from thirdparty.mtcnn import FaceDetector
except Exception:
    from easycv.thirdparty.mtcnn import FaceDetector


@PREDICTORS.register_module()
class TorchYoloXPredictor(PredictorInterface):

    def __init__(self,
                 model_path,
                 max_det=100,
                 score_thresh=0.5,
                 model_config=None):
        """
        init model

        Args:
          model_path: model file path
          max_det: maximum number of detection
          score_thresh:  score_thresh to filter box
          model_config: config string for model to init, in json format
        """
        self.model_path = model_path
        self.max_det = max_det
        if model_config:
            model_config = json.loads(model_config)
        else:
            model_config = {}
        self.score_thresh = model_config[
            'score_thresh'] if 'score_thresh' in model_config else score_thresh

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

        test_pipeline = self.cfg.test_pipeline

        self.CLASSES = self.cfg.CLASSES

        # build pipeline
        pipeline = [build_from_cfg(p, PIPELINES) for p in test_pipeline]
        self.pipeline = Compose(pipeline)

    def predict(self, input_data_list, batch_size=-1):
        """
    using session run predict a number of samples using batch_size

    Args:
      input_data_list:  a list of numpy array(in rgb order), each array is a sample
        to be predicted
      batch_size: batch_size passed by the caller, you can also ignore this param and
        use a fixed number if you do not want to adjust batch_size in runtime
    Return:
      result: a list of dict, each dict is the prediction result of one sample
        eg, {"output1": value1, "output2": value2}, the value type can be
        python int str float, and numpy array
    """
        output_list = []
        for idx, img in enumerate(input_data_list):
            if type(img) is not np.ndarray:
                img = np.asarray(img)

            ori_img_shape = img.shape[:2]
            data_dict = {
                'ori_img_shape': ori_img_shape,
                'img': cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            }
            data_dict = self.pipeline(data_dict)
            img = data_dict['img']
            img = torch.unsqueeze(img._data, 0).to(self.device)
            data_dict.pop('img')
            det_out = self.model(
                img, mode='test', img_metas=[data_dict['img_metas']._data])
            # det_out = det_out[:self.max_det]
            # scale box to original image scale, this logic has some operation
            # that can not be traced, see
            # https://discuss.pytorch.org/t/windows-libtorch-c-load-cuda-module-with-std-runtime-error-message-shape-4-is-invalid-for-input-if-size-40/63073/4
            # det_out = scale_coords(img.shape[2:], det_out, ori_img_shape, (scale_factor, pad))

            detection_scores = det_out['detection_scores'][0]
            sel_ids = detection_scores > self.score_thresh
            detection_boxes = det_out['detection_boxes'][0][sel_ids]
            detection_classes = det_out['detection_classes'][0][sel_ids]
            num_boxes = detection_classes.shape[
                0] if detection_classes is not None else 0
            # print(num_boxes)
            detection_classes_names = [
                self.CLASSES[detection_classes[idx]]
                for idx in range(num_boxes)
            ]

            out = {
                'ori_img_shape': list(ori_img_shape),
                'detection_boxes': detection_boxes,
                'detection_scores': detection_scores,
                'detection_classes': detection_classes,
                'detection_class_names': detection_classes_names,
            }

            output_list.append(out)

        return output_list


@PREDICTORS.register_module()
class TorchFaceDetector(PredictorInterface):

    def __init__(self, model_path=None, model_config=None):
        """
    init model, add a facedetect and align for img input.

    Args:
      model_path: model file path
      model_config: config string for model to init, in json format
    """

        self.detector = FaceDetector()

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

    def predict(self, input_data_list, batch_size=-1, threshold=0.95):
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
    Raise:
       if detect !=1 face in a img, then do nothing for this image
    """
        num_image = len(input_data_list)
        assert len(
            input_data_list) > 0, 'input images should not be an empty list'

        image_list = input_data_list
        outputs_list = []

        for idx, img in enumerate(image_list):
            if type(img) is not np.ndarray:
                img = np.asarray(img)

            ori_img_shape = img.shape[:2]
            bbox, ld = self.detector.safe_detect(img)
            _scores = np.array([i[-1] for i in bbox])

            boxes = []
            scores = []
            for idx, s in enumerate(_scores):
                if s > threshold:
                    boxes.append(bbox[idx][:-1])
                    scores.append(bbox[idx][-1])
            boxes = np.array(boxes)
            scores = np.array(scores)

            out = {
                'ori_img_shape': list(ori_img_shape),
                'detection_boxes': boxes,
                'detection_scores': scores,
                'detection_classes': [0] * boxes.shape[0],
                'detection_class_names': ['face'] * boxes.shape[0],
            }

            outputs_list.append(out)

        return outputs_list


@PREDICTORS.register_module()
class TorchYoloXClassifierPredictor(PredictorInterface):

    def __init__(self,
                 models_root_dir,
                 max_det=100,
                 cls_score_thresh=0.01,
                 det_model_config=None,
                 cls_model_config=None):
        """
    init model, add a yolox and classification predictor for img input.

    Args:
      models_root_dir: models_root_dir/detection/*.pth and models_root_dir/classification/*.pth
      det_model_config: config string for detection model to init, in json format
      cls_model_config: config string for classification model to init, in json format
    """
        det_model_path = glob(
            '%s/detection/*.pt*' % models_root_dir, recursive=True)
        assert (len(det_model_path) == 1)
        cls_model_path = glob(
            '%s/classification/*.pt*' % models_root_dir, recursive=True)
        assert (len(cls_model_path) == 1)

        self.det_predictor = TorchYoloXPredictor(
            det_model_path[0], max_det=max_det, model_config=det_model_config)
        self.cls_predictor = TorchClassifier(
            cls_model_path[0], model_config=cls_model_config)
        self.cls_score_thresh = cls_score_thresh

    def predict(self, input_data_list, batch_size=-1):
        """
    using session run predict a number of samples using batch_size

    Args:
      input_data_list:  a list of numpy array(in rgb order), each array is a sample
        to be predicted
      batch_size: batch_size passed by the caller, you can also ignore this param and
        use a fixed number if you do not want to adjust batch_size in runtime
    Return:
      result: a list of dict, each dict is the prediction result of one sample
        eg, {"output1": value1, "output2": value2}, the value type can be
        python int str float, and numpy array
    """
        results = self.det_predictor.predict(
            input_data_list, batch_size=batch_size)

        for img_idx, img in enumerate(input_data_list):
            detection_boxes = results[img_idx]['detection_boxes']
            detection_classes = results[img_idx]['detection_classes']
            detection_scores = results[img_idx]['detection_scores']

            crop_img_batch = []
            for idx in range(detection_boxes.shape[0]):
                xyxy = [int(a) for a in detection_boxes[idx]]
                cropImg = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                crop_img_batch.append(cropImg)

            if len(crop_img_batch) > 0:
                cls_output = self.cls_predictor.predict(
                    crop_img_batch, batch_size=32)
            else:
                cls_output = []

            class_name_list = []
            class_id_list = []
            class_score_list = []
            det_bboxes = []
            product_count_dict = {}

            for idx in range(len(cls_output)):
                class_name = cls_output[idx]['class_name'][0]
                class_score = cls_output[idx]['class_probs'][class_name]
                if class_score < self.cls_score_thresh:
                    continue

                if class_name not in product_count_dict:
                    product_count_dict[class_name] = 1
                else:
                    product_count_dict[class_name] += 1
                class_name_list.append(class_name)
                class_id_list.append(int(cls_output[idx]['class'][0]))
                class_score_list.append(class_score)
                det_bboxes.append([float(a) for a in detection_boxes[idx]])

            results[img_idx].update({
                'detection_boxes': np.array(det_bboxes),
                'detection_scores': class_score_list,
                'detection_classes': class_id_list,
                'detection_class_names': class_name_list,
                'product_count': product_count_dict
            })

        return results
