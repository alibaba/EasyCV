# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
from glob import glob

import numpy as np
import torch
from torchvision.transforms import Compose

from easycv.core.visualization import imshow_bboxes
from easycv.datasets.registry import PIPELINES
from easycv.datasets.utils import replace_ImageToTensor
from easycv.file import io
from easycv.models import build_model
from easycv.models.detection.utils import postprocess
from easycv.thirdparty.mtcnn import FaceDetector
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.constant import CACHE_DIR
from easycv.utils.misc import deprecated
from easycv.utils.registry import build_from_cfg
from .base import PredictorV2
from .builder import PREDICTORS
from .classifier import TorchClassifier

try:
    from easy_vision.python.inference.predictor import PredictorInterface
except Exception:
    from .interface import PredictorInterface


@PREDICTORS.register_module()
class DetectionPredictor(PredictorV2):
    """Generic Detection Predictor, it will filter bbox results by ``score_threshold`` .
    """

    def __init__(self,
                 model_path,
                 config_file=None,
                 batch_size=1,
                 device=None,
                 save_results=False,
                 save_path=None,
                 pipelines=None,
                 score_threshold=0.5,
                 *arg,
                 **kwargs):
        super(DetectionPredictor, self).__init__(
            model_path,
            config_file=config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            pipelines=pipelines,
        )
        self.score_thresh = score_threshold
        self.CLASSES = self.cfg.get('CLASSES', None)

    def build_processor(self):
        if self.pipelines is not None:
            pipelines = self.pipelines
        elif self.cfg is None:
            pipelines = []
        else:
            pipelines = self.cfg.get('test_pipeline', [])

        # for batch inference
        self.pipelines = replace_ImageToTensor(pipelines)

        return super().build_processor()

    def postprocess_single(self, inputs, *args, **kwargs):
        if inputs['detection_scores'] is None or len(
                inputs['detection_scores']) < 1:
            return inputs

        scores = inputs['detection_scores']
        if scores is not None and self.score_thresh > 0:
            keeped_ids = scores > self.score_thresh
            inputs['detection_scores'] = inputs['detection_scores'][keeped_ids]
            inputs['detection_boxes'] = inputs['detection_boxes'][keeped_ids]
            inputs['detection_classes'] = inputs['detection_classes'][
                keeped_ids]

        class_names = []
        for _, classes_id in enumerate(inputs['detection_classes']):
            if classes_id is None:
                class_names.append(None)
            elif self.CLASSES is not None and len(self.CLASSES) > 0:
                class_names.append(self.CLASSES[int(classes_id)])
            else:
                class_names.append(classes_id)

        inputs['detection_class_names'] = class_names

        return inputs

    def visualize(self, img, results, show=False, out_file=None):
        """Only support show one sample now."""
        bboxes = results['detection_boxes']
        labels = results['detection_class_names']
        img = self._load_input(img)['img']
        imshow_bboxes(
            img,
            bboxes,
            labels=labels,
            colors='cyan',
            text_color='cyan',
            font_size=18,
            thickness=2,
            font_scale=0.0,
            show=show,
            out_file=out_file)


class _JitProcessorWrapper:

    def __init__(self, processor, device) -> None:
        self.processor = processor
        self.device = device

    def __call__(self, results):
        if self.processor is not None:
            from mmcv.parallel import DataContainer as DC
            outputs = {}
            img = results['img']
            img = torch.from_numpy(img).to(self.device)
            img, img_meta = self.processor(img.unsqueeze(0))  # process batch
            outputs['img'] = DC(
                img.squeeze(0),
                stack=True)  # DC wrapper for collate batch and to device
            outputs['img_metas'] = DC(img_meta, cpu_only=True)
            return outputs
        return results


@PREDICTORS.register_module()
class YoloXPredictor(DetectionPredictor):
    """Detection predictor for Yolox."""

    def __init__(self,
                 model_path,
                 config_file=None,
                 batch_size=1,
                 use_trt_efficientnms=False,
                 device=None,
                 save_results=False,
                 save_path=None,
                 pipelines=None,
                 max_det=100,
                 score_thresh=0.5,
                 nms_thresh=None,
                 test_conf=None,
                 *arg,
                 **kwargs):
        self.max_det = max_det
        self.use_trt_efficientnms = use_trt_efficientnms

        if model_path.endswith('jit'):
            self.model_type = 'jit'
        elif model_path.endswith('blade'):
            self.model_type = 'blade'
        else:
            self.model_type = 'raw'

        if self.model_type == 'blade' or self.use_trt_efficientnms:
            import torch_blade

        if self.model_type != 'raw' and config_file is None:
            config_file = model_path + '.config.json'

        super(YoloXPredictor, self).__init__(
            model_path,
            config_file=config_file,
            batch_size=batch_size,
            device=device,
            save_results=save_results,
            save_path=save_path,
            pipelines=pipelines,
            score_threshold=score_thresh)

        self.test_conf = test_conf or self.cfg['model'].get('test_conf', 0.01)
        self.nms_thre = nms_thresh or self.cfg['model'].get('nms_thre', 0.65)
        self.CLASSES = self.cfg.get('CLASSES', None) or self.cfg.get(
            'classes', None)
        assert self.CLASSES is not None

    def _build_model(self):
        if self.model_type != 'raw':
            with io.open(self.model_path, 'rb') as infile:
                model = torch.jit.load(infile, self.device)
        else:
            from easycv.utils.misc import reparameterize_models
            model = super()._build_model()
            model = reparameterize_models(model)
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

    def build_processor(self):
        self.jit_preprocess = False
        if self.model_type != 'raw':
            if hasattr(self.cfg, 'export'):
                self.jit_preprocess = self.cfg['export'].get(
                    'preprocess_jit', False)

        if self.model_type != 'raw' and self.jit_preprocess:
            # jit or blade model
            processor = None
            preprocess_path = '.'.join(
                self.model_path.split('.')[:-1] + ['preprocess'])
            if os.path.exists(preprocess_path):
                # use a preprocess jit model to speed up
                with io.open(preprocess_path, 'rb') as infile:
                    processor = torch.jit.load(infile, self.device)
            return _JitProcessorWrapper(processor, self.device)
        else:
            return super().build_processor()

    def forward(self, inputs):
        """Model forward.
        If you need refactor model forward, you need to reimplement it.
        """
        if self.model_type != 'raw':
            with torch.no_grad():
                outputs = self.model(inputs['img'])
                outputs = {'results': outputs}  # convert to dict format
        else:
            outputs = super().forward(inputs)

        if 'img_metas' not in outputs:
            outputs['img_metas'] = inputs['img_metas']

        return outputs

    def post_assign(self, outputs, img_metas):
        detection_boxes = []
        detection_scores = []
        detection_classes = []
        img_metas_list = []

        for i in range(len(outputs)):
            if img_metas:
                img_metas_list.append(img_metas[i])
            if outputs[i].requires_grad == True:
                outputs[i] = outputs[i].detach()

            if outputs[i] is not None:
                bboxes = outputs[i][:, 0:4] if outputs[i] is not None else None
                if img_metas:
                    bboxes /= img_metas[i]['scale_factor'][0]
                detection_boxes.append(bboxes.cpu().numpy())
                detection_scores.append(
                    (outputs[i][:, 4] * outputs[i][:, 5]).cpu().numpy())
                detection_classes.append(outputs[i][:, 6].cpu().numpy().astype(
                    np.int32))
            else:
                detection_boxes.append(None)
                detection_scores.append(None)
                detection_classes.append(None)

        test_outputs = {
            'detection_boxes': detection_boxes,
            'detection_scores': detection_scores,
            'detection_classes': detection_classes,
            'img_metas': img_metas_list
        }
        return test_outputs

    def postprocess_single(self, inputs):
        det_out = inputs
        img_meta = det_out['img_metas']

        if self.model_type != 'raw':
            results = det_out['results']
            if self.use_trt_efficientnms:
                det_out = {}
                det_out['detection_boxes'] = results[1] / img_meta[
                    'scale_factor'][0]
                det_out['detection_scores'] = results[2]
                det_out['detection_classes'] = results[3]
            else:
                if self.model_type == 'jit':
                    det_out = self.post_assign(
                        results.unsqueeze(0), img_metas=[img_meta])
                else:
                    det_out = self.post_assign(
                        postprocess(
                            results.unsqueeze(0), len(self.CLASSES),
                            self.test_conf, self.nms_thre),
                        img_metas=[img_meta])
            det_out['detection_scores'] = det_out['detection_scores'][0]
            det_out['detection_boxes'] = det_out['detection_boxes'][0]
            det_out['detection_classes'] = det_out['detection_classes'][0]

        resuts = super().postprocess_single(det_out)
        resuts['ori_img_shape'] = list(img_meta['ori_img_shape'][:2])
        return resuts


@deprecated(reason='Please use YoloXPredictor.')
@PREDICTORS.register_module()
class TorchYoloXPredictor(YoloXPredictor):

    def __init__(self,
                 model_path,
                 max_det=100,
                 score_thresh=0.5,
                 use_trt_efficientnms=False,
                 model_config=None):
        """
        Args:
          model_path: model file path
          max_det: maximum number of detection
          score_thresh:  score_thresh to filter box
          model_config: config string for model to init, in json format
        """
        if model_config:
            if isinstance(model_config, str):
                model_config = json.loads(model_config)
        else:
            model_config = {}

        score_thresh = model_config[
            'score_thresh'] if 'score_thresh' in model_config else score_thresh
        super().__init__(
            model_path,
            config_file=None,
            batch_size=1,
            use_trt_efficientnms=use_trt_efficientnms,
            device=None,
            save_results=False,
            save_path=None,
            pipelines=None,
            max_det=max_det,
            score_thresh=score_thresh,
            nms_thresh=None,
            test_conf=None)

    def predict(self, input_data_list, batch_size=-1, to_numpy=True):
        return super().__call__(input_data_list)


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
